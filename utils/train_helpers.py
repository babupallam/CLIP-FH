import torch
import torch.nn as nn
import clip
from utils.loss.arcface import ArcFace


def unfreeze_image_encoder(clip_model, log):
    for param in clip_model.visual.parameters():
        param.requires_grad = True
    log("CLIP image encoder unfrozen.")


def freeze_image_encoder(clip_model, log):
    for param in clip_model.visual.parameters():
        param.requires_grad = False
    log("CLIP image encoder frozen.")


def freeze_prompt_learner(prompt_learner, log):
    for param in prompt_learner.parameters():
        param.requires_grad = False
    log("Prompt learner frozen.")


def freeze_clip_text_encoder(model, log=None):
    for name, param in model.named_parameters():
        if (
            name.startswith("transformer") or
            "token_embedding" in name or
            "text_projection" in name
        ):
            param.requires_grad = False
    if log:
        log("CLIP text encoder frozen.")


def unfreeze_clip_text_encoder(model, log=None):
    unfrozen = 0
    for name, param in model.named_parameters():
        if (
            name.startswith("transformer") or
            "token_embedding" in name or
            "text_projection" in name
        ):
            param.requires_grad = True
            unfrozen += 1
    if log:
        log(f"CLIP text encoder unfrozen ({unfrozen} layers).")


def freeze_entire_clip_model(model, log=None):
    for param in model.parameters():
        param.requires_grad = False
    if log:
        log("All CLIP parameters frozen.")


from utils.loss.arcface import ArcFace


def build_model(config, freeze_text=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]
    clip_model, _ = clip.load("ViT-B/16" if model_name == "vitb16" else "RN50", device=device)

    if freeze_text:
        freeze_clip_text_encoder(clip_model)

    image_embed_dim = clip_model.visual.output_dim
    num_classes = config["num_classes"]
    classifier = nn.Linear(image_embed_dim, num_classes)
    return clip_model, classifier


def register_bnneck_and_arcface(model, feat_dim, num_classes, device, logger=None):
    model.bottleneck = nn.BatchNorm1d(feat_dim).to(device)
    model.bottleneck.bias.requires_grad = False

    model.arcface = ArcFace(
        in_features=feat_dim,
        out_features=num_classes,
        s=30.0,
        m=0.5,
    ).to(device)

    if logger:
        logger("BNNeck and ArcFace head registered.")


# train_helpers.py
import torch.nn as nn
from engine.prompt_learner import TextualInversionMLP
from utils.clip_patch import MultiModalInteraction

def build_promptsg_models(config, num_classes, device):
    pseudo_dim = config['pseudo_token_dim']
    transformer_layers = config['transformer_layers']

    inversion_model = TextualInversionMLP(pseudo_dim, pseudo_dim).to(device)
    multimodal_module = MultiModalInteraction(dim=pseudo_dim, depth=transformer_layers).to(device)
    classifier = nn.Linear(pseudo_dim, num_classes).to(device)

    return inversion_model, multimodal_module, classifier



import clip
import torch

def compose_prompt(text_encoder, pseudo_token_embedding, templates=("A detailed photo of a", "hand."), device="cuda"):
    """
    Compose prompts using prefix + pseudo-token + suffix.

    Args:
        text_encoder: CLIP text encoder
        pseudo_token_embedding: [B, D] tensor
        templates: Tuple of (prefix, suffix)
        device: device to send tokens to

    Returns:
        Tensor of shape [B, 3, D]
    """
    batch_size = pseudo_token_embedding.shape[0]
    prefix_tokens = clip.tokenize([templates[0]] * batch_size).to(device)
    suffix_tokens = clip.tokenize([templates[1]] * batch_size).to(device)

    with torch.no_grad():
        prefix_emb = text_encoder(prefix_tokens).float()
        suffix_emb = text_encoder(suffix_tokens).float()

    composed = torch.stack([prefix_emb, pseudo_token_embedding, suffix_emb], dim=1)  # [B, 3, D]
    return composed
