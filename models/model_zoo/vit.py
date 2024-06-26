#!/usr/bin/env python3
"""Vision Transformer."""
import os
import sys
import torch

from timm.models import create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, '../'))

# import backbones  # noqa: F401

def ViT_B_21K(args):
    """Construct vit_b_p16_224 pretrained on ImageNet-21K."""
    model = create_model(
        'jx_vit_base_patch16_224_in21k', # call from @register_model
        pretrained=False,
        num_classes=21843,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    return _load_checkpoint(args, model)


def _load_checkpoint(args, model):
    """Load the checkpoint into the given model."""
    if args.pretrained_model == "vit-b-22k":
        # path = os.path.join(ROOT_DIR, "../checkpoints/vit_base_p16_224_in22k.pth")
        # path = os.path.join(ROOT_DIR, "../checkpoints_model/vit_base_p16_224_in22k.pth")
        # path = os.path.join(ROOT_DIR, "../checkpoints_model/vit_base_p16_224_in22k.pth")
        path = "/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/lmeraser/models/model_zoo/checkpoints_model/vit_base_p16_224_in22k.pth" # change this path as per your usage

    else:
        raise NotImplementedError
    checkpoint = torch.load(path, map_location="cpu")

    if "module" in checkpoint:
        checkpoint = checkpoint["module"]
    # for key in list(checkpoint.keys()):
    #     if key in ["pre_logits.fc.bias", "pre_logits.fc.weight"]: # ["head.bias", "head.weight"]:
    #         del checkpoint[key]

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model