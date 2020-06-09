import torch
import torchvision
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import PIL
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# NEW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

# NEW
def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

# VOCSegmentation returns a raw dataset: images are non-resized and in the PIL format. To transform them
# something suitable for input to PyTorch, we need to wrap the output in our own dataset class.
class PascalVOCSegmentationDataset(Dataset):
    def __init__(self, raw):
        super().__init__()
        self._dataset = raw
        self.resize_img = torchvision.transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR)
        self.resize_segmap = torchvision.transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST)
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        img, segmap = self._dataset[idx]
        img, segmap = self.resize_img(img), self.resize_segmap(segmap)
        img, segmap = np.array(img), np.array(segmap)
        img, segmap = (img / 255).astype('float32'), segmap.astype('int32')
        img = np.transpose(img, (-1, 0, 1))

        # The PASCAL VOC dataset PyTorch provides labels the edges surrounding classes in 255-valued
        # pixels in the segmentation map. However, PyTorch requires class values to be contiguous
        # in range 0 through n_classes, so we must relabel these pixels to 21.
        segmap[segmap == 255] = 21
        
        return img, segmap

def get_dataloader(rank, world_size):
    _PascalVOCSegmentationDataset = torchvision.datasets.VOCSegmentation(
        '/mnt/pascal_voc_segmentation/', year='2012', image_set='train', download=True,
        transform=None, target_transform=None, transforms=None
    )
    dataset = PascalVOCSegmentationDataset(_PascalVOCSegmentationDataset)
    
    # NEW
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, sampler=sampler)
    
    return dataloader

# num_classes is 22. PASCAL VOC includes 20 classes of interest, 1 background class, and the 1
# special border class mentioned in the previous comment. 20 + 1 + 1 = 22.
def get_model():
    return torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, progress=True, num_classes=22, aux_loss=None
    )

def train(rank, num_epochs, world_size):
    # NEW
    init_process(rank, world_size)
    print(f"Rank {rank}/{world_size} training process initialized.\n")

    # NEW
    # Since this is a single-instance multi-GPU training script, it's important that only one
    # process handle downloading of the data, to:
    #
    # * Avoid race conditions implicit in having multiple processes attempt to write to the same
    #   file simultaneously.
    # * Avoid downloading the data in multiple processes simultaneously.
    #
    # Since the data is cached on disk, we can construct and discard the dataloader and model in
    # the master process only to get the data. The other processes are held back by the barrier.
    if rank == 0:
        get_dataloader(rank, world_size)
        get_model()
    dist.barrier()
    print(f"Rank {rank}/{world_size} training process passed data download barrier.\n")

    model = get_model()
    model.cuda(rank)
    model.train()
        
    # NEW
    model = DistributedDataParallel(model, device_ids=[rank])
    
    dataloader = get_dataloader(rank, world_size)
    
    # since the background class doesn't matter nearly as much as the classes of interest to the
    # overall task a more selective loss would be more appropriate, however this training script
    # is merely a benchmark so we'll just use simple cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # NEW
    # Since we are computing the average of several batches at once (an effective batch size of
    # world_size * batch_size) we scale the learning rate to match.
    optimizer = Adam(model.parameters(), lr=1e-3 * world_size)
    
    writer = SummaryWriter(f'/spell/tensorboards/model_2')
        
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for i, (batch, segmap) in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.cuda(rank)
            segmap = segmap.cuda(rank)

            output = model(batch)['out']
            loss = criterion(output, segmap.type(torch.int64))
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if i % 10 == 0:
                print(
                    f'Finished epoch {epoch}, rank {rank}/{world_size}, batch {i}. '
                    f'Loss: {curr_loss:.3f}.\n'
                )
            if rank == 0:
                writer.add_scalar('training loss', curr_loss)
            losses.append(curr_loss)

        print(
            f'Finished epoch {epoch}, rank {rank}/{world_size}. '
            f'Avg Loss: {np.mean(losses)}; Median Loss: {np.min(losses)}.\n'
        )
        
        if rank == 0:
            if not os.path.exists('/spell/checkpoints/'):
                os.mkdir('/spell/checkpoints/')
            torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}.pth')

# NEW
NUM_EPOCHS = 20
WORLD_SIZE = torch.cuda.device_count()
def main():
    mp.spawn(train,
        args=(NUM_EPOCHS, WORLD_SIZE),
        nprocs=WORLD_SIZE,
        join=True)

if __name__=="__main__":
    main()
