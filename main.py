import torch
import clip

def freeze_clip_text_encoder(model):
    """
    Utility to freeze all CLIP text encoder parameters (transformer, token_embedding, text_projection).
    """
    print("[INFO] Re-freezing text encoder parameters as per checkpoint metadata...")
    for name, param in model.named_parameters():
        if (
            name.startswith("transformer.") or
            "token_embedding" in name or
            "text_projection" in name
        ):
            param.requires_grad = False
            print(f" Froze {name}")

def check_text_encoder_frozen(checkpoint_path, model_name="ViT-B/16", device="cpu"):
    # Load base CLIP model architecture
    model, _ = clip.load(model_name, device=device)

    # Load full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Re-freeze if checkpoint metadata says it was frozen
    if checkpoint.get("metadata", {}).get("freeze_text_encoder", False):
        freeze_clip_text_encoder(model)

    # Gather text encoder parameters
    text_params = [
        (n, p) for n, p in model.named_parameters()
        if n.startswith("transformer.") or "token_embedding" in n or "text_projection" in n
    ]

    # Check frozen status
    frozen = all(not p.requires_grad for _, p in text_params)
    print(f"\n[] Text encoder is {'FROZEN ' if frozen else 'NOT frozen '}")

    print("\n Breakdown:")
    for name, param in text_params:
        print(f"{'' if param.requires_grad else ''} {name}")

# Run it
if __name__ == "__main__":
    checkpoint_path = "saved_models/stage1_stage1_frozen_text_vitb16_11k_dorsal_r_e1_lr00001_bs32_losscross_entropy_BEST.pth"
    check_text_encoder_frozen(checkpoint_path)
