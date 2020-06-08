# deeplab-voc-2012

`pytorch` scripts training DeeplabV3 (trained from scratch) on the PASCAL VOC 2012 Segmentation dataset for 20 epochs. The model architecture and data are taken from the PyTorch pretrained models and example data, respectively (`torchvision.models.segmentation.deeplabv3_resnet101` and `torchvision.datasets.VOCSegmentation`). This training script is useful for benchmarking large-scale CNN training jobs.

To run on Spell:

```bash
spell run --machine-type v100 \
  --github-url https://github.com/ResidentMario/spell-deeplab-voc-2012.git \
  "python models/1_initial_model.py"
```
```bash
spell run --machine-type v100x4 \
  --github-url https://github.com/ResidentMario/spell-deeplab-voc-2012.git \
  "python models/2_pytorch_distributed_model.py"
```
```bash
spell run --machine-type v100x8 \
  --github-url https://github.com/ResidentMario/spell-deeplab-voc-2012.git \
  "python models/2_pytorch_distributed_model.py"
```
