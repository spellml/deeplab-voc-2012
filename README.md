# deeplab-voc-2012 <a href="https://web.spell.ml/workspace_create?workspaceName=deeplab-voc-2012&githubUrl=https%3A%2F%2Fgithub.com%2Fspellml%2Fdeeplab-voc-2012"><img src=https://spell.ml/badge.svg height=20px/></a>

`pytorch` scripts training DeeplabV3 (trained from scratch) on the PASCAL VOC 2012 Segmentation dataset for 20 epochs. The model architecture and data are taken from the PyTorch pretrained models and example data, respectively (`torchvision.models.segmentation.deeplabv3_resnet101` and `torchvision.datasets.VOCSegmentation`). This training script is useful for benchmarking large-scale CNN training jobs.

To run on Spell:

```bash
spell run --machine-type v100 \
  --github-url https://github.com/spellml/deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_1/ \
  "python models/1_initial_model.py"
```
```bash
spell run --machine-type v100x4 \
  --github-url https://github.com/spellml/deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_2/ \
  "python models/2_pytorch_distributed_model.py"
```
```bash
spell run --machine-type v100x8 \
  --github-url https://github.com/spellml/deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_2/ \
  "python models/2_pytorch_distributed_model.py"
```
```bash
spell run --machine-type v100x4 \
  --github-url https://github.com/spellrun/spell-deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_3/ \
  "python models/3_pytorch_distributed_threaded.py"
```
```bash
spell run --machine-type v100x8 \
  --github-url https://github.com/spellml/deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_3/ \
  "python models/3_pytorch_distributed_threaded.py"
```
```bash
spell run --machine-type v100x4 \
  --github-url https://github.com/spellrun/spell-deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_4/ \
  --distributed 1 \
  "python models/4_pytorch_distributed_horovod.py"
```
```bash
spell run --machine-type v100x8 \
  --github-url https://github.com/spellml/deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_4/ \
  --distributed 1 \
  "python models/4_pytorch_distributed_horovod.py"
```
```bash
spell run --machine-type v100 \
  --github-url https://github.com/spellml/deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_4/ \
  --distributed 4 \
  "python models/4_pytorch_distributed_horovod.py"
```
```bash
spell run --machine-type v100 \
  --github-url https://github.com/spellml/deeplab-voc-2012.git \
  --tensorboard-dir /spell/tensorboards/model_4/ \
  --distributed 8 \
  "python models/4_pytorch_distributed_horovod.py"
```
