# models/clip_patch.py

import clip


def load_clip_with_patch(model_type, device, freeze_all=True):
    model_map = {
        "vitb16": "ViT-B/16",
        "vitb32": "ViT-B/32",
        "rn50": "RN50",
        "rn101": "RN101",
        "rn50x4": "RN50x4",
        "rn50x16": "RN50x16",
        "rn50x64": "RN50x64"
    }

    model_name = model_map.get(model_type.lower())
    if model_name is None:
        raise ValueError(f"❌ Unknown model type: {model_type}")

    model, _ = clip.load(model_name, device=device)

    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False

    return model, _
