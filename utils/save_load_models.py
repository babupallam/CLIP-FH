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
    metrics_log=None,
    prompt_learner=None
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
    text_encoder_frozen = config.get("freeze_text", False)

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

    # Save Prompt Learner separately if available
    if prompt_learner is not None:
        prompt_path = path.replace(".pth", "_prompt.pth")
        torch.save(prompt_learner.state_dict(), prompt_path)
        print(f"Saved Prompt Learner to: {prompt_path}")

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
        "rng_state": rng_state,
        "trainable_flags": {
            n: p.requires_grad for n, p in model.named_parameters()
        }
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




import torch
import os
import warnings

def load_checkpoint(
    path,
    model,
    classifier=None,
    optimizer=None,
    scheduler=None,
    device="cpu",
    config=None
):
    """
    Universal checkpoint loader for CLIP-FH training/eval stages.

    Args:
        path (str): Path to the saved model checkpoint (.pth).
        model (nn.Module): The CLIP model instance.
        classifier (nn.Module, optional): The classifier head (Stage 1).
        optimizer (Optimizer, optional): The optimizer used during training.
        scheduler (Scheduler, optional): LR scheduler (if used).
        device (str): 'cuda' or 'cpu'.
        config (dict, optional): Runtime config for comparison/logging.

    Returns:
        checkpoint (dict)
        checkpoint_config (dict)
        epoch (int)
        metadata (dict)
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # === Load model weights ===
    if "model_state_dict" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing:
            warnings.warn(f"[WARN] Missing model keys: {missing}")
        if unexpected:
            warnings.warn(f"[WARN] Unexpected model keys: {unexpected}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    # === Restore requires_grad flags ===
    if "trainable_flags" in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint["trainable_flags"]:
                param.requires_grad = checkpoint["trainable_flags"][name]
    else:
        warnings.warn("[INFO] No trainable_flags found in checkpoint.")

    # === Load classifier (if present) ===
    if classifier and "classifier_state_dict" in checkpoint:
        try:
            classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=False)
        except Exception as e:
            warnings.warn(f"[WARN] Failed to load classifier: {e}")
    elif classifier is not None:
        warnings.warn("[INFO] Classifier passed in, but no classifier_state_dict found in checkpoint.")

    # === Load optimizer (if present) ===
    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            warnings.warn(f"[WARN] Failed to load optimizer state: {e}")

    # === Load scheduler (if present) ===
    if scheduler and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            warnings.warn(f"[WARN] Failed to load scheduler state: {e}")

    # === RNG restoration ===
    if "rng_state" in checkpoint:
        torch_rng_state = checkpoint["rng_state"].get("torch_rng_state", None)
        if torch_rng_state is not None:
            torch_rng_state = torch.tensor(torch_rng_state, dtype=torch.uint8, device="cpu").contiguous()
            torch.set_rng_state(torch_rng_state)

        if torch.cuda.is_available():
            cuda_rng_state = checkpoint["rng_state"].get("cuda_rng_state", None)
            if cuda_rng_state is not None:
                cuda_rng_state_tensor_list = []
                for i, state in enumerate(cuda_rng_state):
                    print(f"[DEBUG] cuda_rng_state[{i}] - type: {type(state)}, dtype: {getattr(state, 'dtype', 'n/a')}")
                    if not isinstance(state, torch.ByteTensor):
                        print(f"[FIX] Converting cuda_rng_state[{i}] to torch.ByteTensor on CUDA")
                        state = torch.tensor(state, dtype=torch.uint8, device="cpu")

                    cuda_rng_state_tensor_list.append(state)
                print(f"[INFO] Setting RNG state for {len(cuda_rng_state_tensor_list)} CUDA devices.")

                for i, s in enumerate(cuda_rng_state_tensor_list):
                    print(f"[CONFIRM] Tensor[{i}] = {type(s)}, device: {s.device}, dtype: {s.dtype}")

                torch.cuda.set_rng_state_all(cuda_rng_state_tensor_list)

    # === Meta info ===
    checkpoint_config = checkpoint.get("config", {})
    metadata = checkpoint.get("metadata", {})
    epoch = checkpoint.get("epoch", 0)

    print(f"Checkpoint loaded from: {path}")
    print(f"Restored to epoch: {epoch}")

    # === Optional config comparison ===
    if config and checkpoint_config and checkpoint_config != config:
        warnings.warn("[INFO] Loaded checkpoint config differs from current config.")

    return checkpoint, checkpoint_config, epoch, metadata
