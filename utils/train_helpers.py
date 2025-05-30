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


def unfreeze_entire_clip_image_encoder(clip_model, logger_fn=None):
    for name, param in clip_model.visual.named_parameters():
        param.requires_grad = True
        if logger_fn:
            logger_fn(f"[Trainable] {name}")


def unfreeze_clip_text_encoder(model, logger=print):
    for name, param in model.named_parameters():
        if "transformer" in name or "ln_final" in name or "token_embedding" in name:
            param.requires_grad = True
    logger("CLIP text encoder unfrozen.")

def unfreeze_clip_image_encoder(model, logger=None, unfreeze_blocks=0):
    """
    Unfreezes the last N blocks of the CLIP visual encoder.
    Supports both ViT (resblocks) and RN50 (layer1-4) backbones.
    """

    # Freeze all visual parameters
    for param in model.visual.parameters():
        param.requires_grad = False

    # === ViT-based CLIP ===
    if hasattr(model.visual, "transformer") and hasattr(model.visual.transformer, "resblocks"):
        resblocks = model.visual.transformer.resblocks
        total_blocks = len(resblocks)
        if unfreeze_blocks > 0:
            start = max(0, total_blocks - unfreeze_blocks)
            for idx in range(start, total_blocks):
                for param in resblocks[idx].parameters():
                    param.requires_grad = True
            if logger:
                logger(f"[ViT] Unfroze {unfreeze_blocks} resblocks: {start}{total_blocks - 1}")
        else:
            if logger:
                logger("[ViT] All visual blocks kept frozen")

    # === RN-based CLIP ===
    elif all(hasattr(model.visual, layer) for layer in ["layer1", "layer2", "layer3", "layer4"]):
        rn_layers = ["layer1", "layer2", "layer3", "layer4"]
        layers_to_unfreeze = rn_layers[-unfreeze_blocks:] if unfreeze_blocks > 0 else []

        for name, module in model.visual.named_children():
            if name in layers_to_unfreeze:
                for param in module.parameters():
                    param.requires_grad = True

        if logger:
            logger(f"[RN50] Unfroze layers: {', '.join(layers_to_unfreeze) if layers_to_unfreeze else 'none'}")
    else:
        raise ValueError("Unknown or unsupported visual architecture.")


def freeze_entire_clip_model(model, log=None):
    for param in model.parameters():
        param.requires_grad = False
    if log:
        log("All CLIP parameters frozen.")


from .train_helpers import freeze_clip_text_encoder  # make sure this exists
from utils.loss.arcface import ArcFace  # Only needed for v5

def build_model(config, freeze_text=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP backbone
    model_name = config["model"].lower()
    clip_model, _ = clip.load("ViT-B/16" if model_name == "vitb16" else "RN50", device=device)

    # v6 logic: reinitialize visual encoder if specified
    if config.get("clip_init", "default") == "random":
        print("[INFO] Reinitializing CLIP visual encoder with random weights (v6)")
        reinitialize_clip_visual(clip_model)

    if freeze_text:
        freeze_clip_text_encoder(clip_model)

    image_embed_dim = clip_model.visual.output_dim
    num_classes = config["num_classes"]

    # Choose classifier type from config
    classifier_type = config.get("classifier", "linear").lower()

    # === V1: Basic Linear ===
    if classifier_type == "linear":
        print("classifier is linear")
        classifier = nn.Linear(image_embed_dim, num_classes)
        nn.init.xavier_normal_(classifier.weight)
        if classifier.bias is not None:
            nn.init.zeros_(classifier.bias)

    # === V4: Sequential with BN + Dropout + Linear ===
    elif classifier_type == "sequential":

        print("classifier is sequential")
        classifier = nn.Sequential(
            nn.BatchNorm1d(image_embed_dim),
            nn.Dropout(p=0.2),
            nn.Linear(image_embed_dim, num_classes)
        )
        # Initialize only the Linear layer
        for layer in classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    # === V5: ArcFace ===
    elif classifier_type == "arcface":
        print("classifier is arcface")
        classifier = ArcFace(in_features=image_embed_dim, out_features=num_classes)

    else:
        raise ValueError(f"[ERROR] Unknown classifier type: {classifier_type}")

    return clip_model, classifier


def reinitialize_clip_visual(clip_model):
    """
    Re-initializes CLIP's visual encoder weights from scratch using Kaiming Normal init.
    Useful for experiments without pretraining.
    """
    for name, module in clip_model.visual.named_modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


import torch.nn as nn
from utils.loss.arcface import ArcFace

def register_bnneck_and_arcface(model, config, feat_dim, num_classes, device, logger=None):
    # BNNeck: Normalizes feature vectors before classification
    model.bottleneck = nn.BatchNorm1d(feat_dim).to(device)
    model.bottleneck.bias.requires_grad = False  # Optional, as in CLIP-ReID

    # ArcFace: Margin-based head for discriminative features
    # v2
    """
    model.arcface = ArcFace(
        in_features=feat_dim,
        out_features=num_classes,
        s=30.0,
        m=0.5,
    ).to(device)
    """
    # v3
    arcface_scale = config.get("arcface_scale", 30.0)
    arcface_margin = config.get("arcface_margin", 0.5)
    model.arcface = ArcFace(
        in_features=feat_dim,
        out_features=num_classes,
        s=arcface_scale,
        m=arcface_margin
    ).to(device)


    # Standard linear classifier (optional but helps early-stage stability)
    model.classifier = nn.Linear(feat_dim, num_classes).to(device)

    # Optionally initialize classifier weights
    nn.init.normal_(model.classifier.weight, std=0.001)
    nn.init.constant_(model.classifier.bias, 0)

    if logger:
        logger(f"BNNeck and ArcFace head registered (scale={arcface_scale}, margin={arcface_margin})")

# train_helpers.py

# Import required modules
import torch.nn as nn
from engine.prompt_learner import TextualInversionMLP  # Pseudo-token generator
from utils.clip_patch import MultiModalInteraction      # Cross-attention fusion module
from utils.loss.arcface import ArcFace                  # ArcFace classification head

def build_promptsg_models(config, num_classes, device):
    #  Map each CLIP model type to its image feature output dimension
    model_dim_map = {
        "vitb16": 512,
        "vitb32": 512,
        "rn50": 1024,
        "rn101": 512,
        "rn50x4": 640,
        "rn50x16": 768,
        "rn50x64": 1024
    }

    # Get model name from config and look up its image feature dimension
    clip_model_name = config["model"].lower()
    pseudo_dim = model_dim_map.get(clip_model_name, 512)  # Use 512 if model is unknown

    # Number of transformer layers to use in the multimodal module (from config)
    transformer_layers = config['transformer_layers']

    # === Build PromptSG modules ===

    # 1. Pseudo-token generator (takes image features  outputs token for prompt)
    inversion_model = TextualInversionMLP(pseudo_dim, pseudo_dim).to(device)

    # 2. Cross-attention fusion block (fuses prompt tokens + image features)
    multimodal_module = MultiModalInteraction(dim=pseudo_dim, depth=transformer_layers).to(device)

    # === BNNeck Variant (used when classifier is 'arcface') ===
    # BNNeck is a normalization layer (BatchNorm1d) placed before the classifier.
    # It helps to stabilize features and improve the angular margin learning of ArcFace.
    # Optionally, we can also reduce the feature dimensionality before applying BNNeck.

    # Check if config says to use ArcFace classifier
    # ArcFace Classifier:
    # A special classification head that adds an angular margin penalty between classes.
    # It encourages features of the same class to be closer in angle, while pushing
    # different classes further apart. This is especially useful in face/hand/person
    # recognition tasks where intra-class compactness and inter-class separability matter.

    use_bnneck = config.get("classifier", "linear").lower() == "arcface"

    # Optionally apply feature dimension reduction before BNNeck
    use_reduction = config.get("bnneck_reduction", False)
    reduced_dim = config.get("bnneck_dim", 256)  # New dimension if reduced

    if use_bnneck:
        # Reduce features if requested, otherwise pass them unchanged
        if use_reduction:
            reduction = nn.Linear(pseudo_dim, reduced_dim).to(device)
            feat_dim = reduced_dim
        else:
            reduction = nn.Identity().to(device)  # No change
            feat_dim = pseudo_dim

        # BatchNorm neck layer (BNNeck)
        bnneck = nn.BatchNorm1d(feat_dim).to(device)

        # ArcFace classifier head (adds angular margin to improve class separation)
        classifier = ArcFace(in_features=feat_dim, out_features=num_classes).to(device)

        # Return all modules
        return inversion_model, multimodal_module, reduction, bnneck, classifier

    else:
        # === Simple Linear Classifier (used when classifier is not 'arcface') ===
        classifier = nn.Linear(pseudo_dim, num_classes).to(device)

        # Return only modules needed for linear path (reduction and bnneck = None)
        return inversion_model, multimodal_module, None, None, classifier

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
