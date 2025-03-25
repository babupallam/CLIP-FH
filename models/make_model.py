import torch
import clip
import torch.nn as nn


def build_model(config, freeze_text=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]
    clip_model, _ = clip.load("ViT-B/16" if model_name == "vitb16" else "RN50", device=device)

    if freeze_text:
        for p in clip_model.transformer.parameters():
            p.requires_grad = False

    image_embed_dim = clip_model.visual.output_dim
    num_classes = config["num_classes"]

    classifier = nn.Linear(image_embed_dim, num_classes)

    return clip_model, classifier
