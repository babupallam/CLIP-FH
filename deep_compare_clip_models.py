import torch
import clip
from pathlib import Path
from datetime import datetime

# Setup
model_paths = {
    "official_clip": None,
    "stage1_frozen_text": "saved_models/stage1_frozen_text_vitb16_11k_dorsal_r_e30_lr0p0001_bs32_cross_entropy_BEST.pth",
    "stage2_joint": "saved_models/stage2_joint_vitb16_11k_dorsal_r_vitb16_11k_dorsal_r_finetune_e0_lr00001_bs32_BEST.pth",
    "stage3_promptsg": "saved_models/stage3_promptsg_promptsg_vitb16_11k_dorsal_r_e20_lr0.0001_bs32_freezeTextTrue_BEST.pth"
}

device = "cpu"
model_name = "ViT-B/16"
log_dir = Path("result_logs")
log_dir.mkdir(exist_ok=True)

def load_model(path=None):
    model, _ = clip.load(model_name, device=device)
    if path:
        state_dict = torch.load(path, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model

def analyze_model(model, ref_params=None):
    summary = {}
    params = list(model.named_parameters())
    buffers = list(model.named_buffers())

    summary["total_params"] = len(params)
    summary["trainable_params"] = sum(p.requires_grad for _, p in params)
    summary["frozen_params"] = summary["total_params"] - summary["trainable_params"]

    summary["image_encoder"] = [n for n, _ in params if n.startswith("visual.")]
    summary["text_encoder"] = [n for n, _ in params if n.startswith("transformer.")]
    summary["prompt_related"] = [n for n, _ in params if "prompt" in n.lower()]
    summary["trainable"] = [n for n, p in params if p.requires_grad]
    summary["buffers"] = buffers

    # Compare parameter names to reference
    current_names = set(n for n, _ in params)
    summary["extra_params"] = sorted(current_names - ref_params) if ref_params else []

    return summary

# Load and analyze
models = {}
analyses = {}

# Load and analyze official first
official_model = load_model()
models["official_clip"] = official_model
analyses["official_clip"] = analyze_model(official_model)
ref_param_names = set(n for n, _ in official_model.named_parameters())

# Load others
for name, path in model_paths.items():
    if name == "official_clip": continue
    m = load_model(path)
    models[name] = m
    analyses[name] = analyze_model(m, ref_params=ref_param_names)

# Write to log
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"model_comparison_{timestamp}.log"

with open(log_path, "w") as f:
    f.write("--- CLIP Model Architecture Comparison Log ---\n\n")
    for name, r in analyses.items():
        f.write(f"Model: {name}\n")
        f.write(f"  Total parameters     : {r['total_params']}\n")
        f.write(f"  Trainable parameters : {r['trainable_params']}\n")
        f.write(f"  Frozen parameters    : {r['frozen_params']}\n")
        f.write(f"  Image encoder params : {len(r['image_encoder'])}\n")
        f.write(f"  Text encoder params  : {len(r['text_encoder'])}\n")
        f.write(f"  Prompt-related params: {len(r['prompt_related'])} ({', '.join(r['prompt_related']) if r['prompt_related'] else 'None'})\n")
        f.write(f"  Buffers              : {len(r['buffers'])}\n")
        f.write(f"  Trainable Preview    : {', '.join(r['trainable'][:5]) + ('...' if len(r['trainable']) > 5 else '')}\n")
        f.write(f"  Extra Parameters     : {len(r['extra_params'])} ({', '.join(r['extra_params']) if r['extra_params'] else 'None'})\n")
        f.write("\n")

    # Summary section
    f.write("--- Summary & Observations ---\n\n")
    for name in model_paths.keys():
        if name == "official_clip":
            continue
        r = analyses[name]
        f.write(f"Model: {name}\n")
        if r["prompt_related"]:
            f.write("  - Prompt learning module detected.\n")
        else:
            f.write("  - No prompt module found.\n")
        if len(r["extra_params"]) > 0:
            f.write("  - Contains custom parameters not in official CLIP.\n")
        else:
            f.write("  - No architectural additions over official CLIP.\n")
        if len(r["trainable"]) < r["total_params"]:
            f.write("  - Freezing strategy was used (partially frozen).\n")
        else:
            f.write("  - All parameters are trainable (no freezing).\n")
        if set(r["trainable"]) == set(analyses["official_clip"]["trainable"]):
            f.write("  - Trainable layers same as official CLIP (check if training was effective).\n")
        f.write("\n")

print(f"[✓] Comprehensive comparison saved to: {log_path}")
