# Modified code of LMEraser with fixed errors
## LMEraser

## Environment Setup

This code has been tested with Python 3.11.5 and PyTorch 2.1.2 with CUDA 12.1 on Ubuntu 22.04. The required packages are listed in `environment.yaml`.

**To set up a conda environment, please follow these steps:**
```bash
conda env create -f environment.yaml -n lmeraser
conda activate lmeraser
```

## Dataset Preparation

Datasets are sourced from torchvision and downloaded automatically. For more details, please refer to [torchvision datasets](https://pytorch.org/vision/stable/datasets.html). The custom datasets directory can be set using `--base_dir` when running the code.

## Pre-trained Model Preparation

The pre-trained vision models used can be downloaded from the provided links and should be placed in `models/checkpoints/`.

### Pre-trained Models
| Backbone  | Pre-trained Objective | Pre-trained Dataset | Download Link | md5sum |
|-----------|-----------------------|---------------------|---------------|--------|
| ViT-B/16  | Supervised            | ImageNet-22k        | [Download](https://drive.google.com/file/d/1zvIqdml4KVArPuWspoHKU7a6e0uAunF8/view?usp=sharing) | -      |
| Swin-B    | Supervised            | ImageNet-22k        | [Download](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) | bf9cc1 |

## Training

```bash
python main.py --test_dataset=cifar100 --loss_filename=cifar100_loss --acc_filename=cifar100_acc
```
## Changable Arguments

Key arguments are listed in `arguments.py`. The default settings are configured for training on CIFAR-100 with a ViT-22k backbone.

### Important Arguments
- `--erasing_method`: Select the erasing method (e.g., `lmeraser`, `random_part_tuning`).
- `--base_dir`: Directory to store datasets.
- `--test_dataset`: Dataset for training.
- `--pretrained_model`: Pre-trained model to use.
- `--one_prompt`: Use one or multiple prompts (default: `False`).
- `--num_gpus`: Number of GPUs to use.
- `--batch_size`: Total batch size.
- `--num_epochs`: Number of training epochs.
- `--lr`: Learning rate.
- `--distance_threshold`: Distance threshold for clustering.

## Acknowledgement

This repository is partially based on [VP](https://github.com/hjbahng/visual_prompting), [VPT](https://github.com/KMnP/vpt), and [DAM-VP](https://github.com/shikiw/DAM-VP). We thank the authors for their impressive works!

## License

This code is released under the MIT License (see LICENSE file for details).
