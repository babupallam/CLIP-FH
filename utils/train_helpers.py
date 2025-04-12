import torch
import torch.nn as nn
import clip

def unfreeze_image_encoder(clip_model, log):
    for param in clip_model.visual.parameters():
        param.requires_grad = True
    log("Unfroze CLIP image encoder.")

def freeze_prompt_learner(prompt_learner, log):
    for param in prompt_learner.parameters():
        param.requires_grad = False
    log("Prompt Learner frozen.")

def freeze_clip_text_encoder(model):
    """
    Freezes all text-related parameters in CLIP (transformer, token embeddings, projection).
    """
    for name, param in model.named_parameters():
        if (
            name.startswith("transformer") or
            "token_embedding" in name or
            "text_projection" in name
        ):
            param.requires_grad = False
            print(f"Freezing {name} parameters")

def build_model(config, freeze_text=False):
    """
    Builds a CLIP-based model for classification.

    Args:
        config (dict): Must include "model" (e.g., "vitb16", "rn50") and "num_classes".
        freeze_text (bool): Whether to freeze the CLIP text encoder.

    Returns:
        clip_model: CLIP backbone
        classifier: Linear classification head
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config["model"]
    clip_model, _ = clip.load("ViT-B/16" if model_name == "vitb16" else "RN50", device=device)

    if freeze_text:
        freeze_clip_text_encoder(clip_model)

    image_embed_dim = clip_model.visual.output_dim
    num_classes = config["num_classes"]
    classifier = nn.Linear(image_embed_dim, num_classes)
    return clip_model, classifier
