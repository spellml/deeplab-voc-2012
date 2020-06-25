import torch
import torchvision
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import PIL
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd

_PascalVOCSegmentationDataset = torchvision.datasets.VOCSegmentation(
    '/mnt/pascal_voc_segmentation/', year='2012', image_set='train', download=True,
    transform=None, target_transform=None, transforms=None
)

# VOCSegmentation returns a raw dataset: images are non-resized and in the PIL format. To transform them
# to something suitable for input to PyTorch, we need to wrap the output in our own dataset class.
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

dataset = PascalVOCSegmentationDataset(_PascalVOCSegmentationDataset)
# NEW
# Distributed sampler.
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=hvd.size(), rank=hvd.rank()
)
dataloader = DataLoader(
    dataset, batch_size=8 * torch.cuda.device_count(), shuffle=False, sampler=sampler
)

# num_classes is 22. PASCAL VOC includes 20 classes of interest, 1 background class, and the 1
# special border class mentioned in the previous comment. 20 + 1 + 1 = 22.
DeepLabV3 = torchvision.models.segmentation.deeplabv3_resnet101(
    pretrained=False, progress=True, num_classes=22, aux_loss=None
)
model = DeepLabV3

# NEW
model = nn.DataParallel(model)

model.cuda()
model.train()

writer = SummaryWriter(f'/spell/tensorboards/model_4')

# since the background class doesn't matter nearly as much as the classes of interest to the
# overall task a more selective loss would be more appropriate, however this training script
# is merely a benchmark so we'll just use simple cross-entropy loss
criterion = nn.CrossEntropyLoss()

def train(NUM_EPOCHS):
    # NEW:
    # set epoch to sampler for shuffling.
    sampler.set_epoch(epoch)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for i, (batch, segmap) in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.cuda()
            segmap = segmap.cuda()

            output = model(batch)['out']
            loss = criterion(output, segmap.type(torch.int64))
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            # if i % 10 == 0:
            #     print(
            #         f'Finished epoch {epoch}, batch {i}. Loss: {curr_loss:.3f}.'
            #     )

            writer.add_scalar(
                'training loss', curr_loss, epoch * len(dataloader) + i
            )
            losses.append(curr_loss)

        # print(
        #     f'Finished epoch {epoch}. '
        #     f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
        # )
        if epoch % 5 == 0:
            if not os.path.exists('/spell/checkpoints/'):
                os.mkdir('/spell/checkpoints/')
            torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}.pth')
    torch.save(model.state_dict(), f'/spell/checkpoints/model_final.pth')

    
if __name__ == '__main__':
    # NEW:
    # Init horovod
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.set_num_threads(1)
    
    # NEW:
    # Scale learning learning rate by size.
    optimizer = optim.Adam(model.parameters(), lr=0.001 * hvd.size())

    # New:
    # Broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # NEW:
    # (optional) Free-ish compression (reduces over-the-wire size -> increases speed).
    compression = hvd.Compression.fp16
    
    # NEW:
    # Wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Average)
    train(20)
