import torch
from datetime import datetime
import hashlib

def save_checkpoint(
    model,
    classifier,
    optimizer,
    config,
    epoch,
    val_metrics,
    path,
    is_best=False,
    scheduler=None,
    train_loss=None,
    metrics_log=None
):
    """
    Save a full training checkpoint supporting all CLIP-FH stages.

    Includes:
    - Full model state_dict (visual, text, prompt if any)
    - Classifier head (if applicable)
    - Optimizer and scheduler
    - Training and validation metrics
    - Full config + metadata
    - Timestamp, config hash, RNG state (for reproducibility)
    """

    # Detect prompt-related parameters
    prompt_params = [n for n, _ in model.named_parameters() if "prompt" in n.lower()]
    has_classifier = classifier is not None
    has_scheduler = scheduler is not None
    clip_variant = config.get("clip_model", "ViT-B/16")
    text_encoder_frozen = config.get("freeze_text", True)

    # Generate config hash for traceability
    config_serialized = str(sorted(config.items()))
    config_hash = hashlib.md5(config_serialized.encode()).hexdigest()

    # Build metadata block
    metadata = {
        "used_prompt_learning": bool(prompt_params),
        "prompt_param_names": prompt_params,
        "freeze_text_encoder": text_encoder_frozen,
        "clip_variant": clip_variant,
        "aspect": config.get("aspect", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "fine_tuned_image_encoder": True,
        "fine_tuned_classifier": has_classifier,
        "has_scheduler": has_scheduler,
        "stage": config.get("stage", "unknown"),
        "save_time": datetime.now().isoformat(),
        "config_hash": config_hash
    }

    # Save RNG states for exact reproducibility
    rng_state = {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

    # Build the checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": val_metrics.get("avg_val_loss"),
        "train_loss": train_loss,
        "top1_accuracy": val_metrics.get("top1_accuracy"),
        "top5_accuracy": val_metrics.get("top5_accuracy"),
        "top10_accuracy": val_metrics.get("top10_accuracy"),
        "config": config,
        "metadata": metadata,
        "rng_state": rng_state
    }

    if has_classifier:
        checkpoint["classifier_state_dict"] = classifier.state_dict()
    if has_scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metrics_log is not None:
        checkpoint["metrics_log"] = metrics_log  # List of per-epoch logs

    torch.save(checkpoint, path)
    tag = "BEST" if is_best else "FINAL"
    print(f"Saved {tag} checkpoint to: {path}")
