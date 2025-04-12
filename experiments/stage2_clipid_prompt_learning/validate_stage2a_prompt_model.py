import os
import sys
import torch
import argparse
import yaml
from tqdm import tqdm
import torch.nn.functional as F

# 🔧 Fix imports for root-level access
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 🔌 Internal modules
from utils.clip_patch import load_clip_with_patch
from engine.prompt_learner import PromptLearner
from utils.dataloaders import get_dataloader


@torch.no_grad()
def validate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")

    # === Load base CLIP model ===
    model_type = config["model"]
    clip_model, _ = load_clip_with_patch(model_type, device=device, freeze_all=True)
    clip_model.eval()

    # === Load dataset ===
    val_dir = config["val_split"]
    loader = get_dataloader(val_dir, batch_size=config["batch_size"], train=False)

    # === Prepare classnames ===
    classnames = loader.dataset.classes
    num_classes = len(classnames)
    print(f"✅ Loaded {len(loader.dataset)} val samples across {num_classes} classes")

    # === Load PromptLearner and weights ===
    prompt_learner = PromptLearner(
        classnames=classnames,
        clip_model=clip_model,
        n_ctx=config.get("n_ctx", 8),
        ctx_init=config.get("ctx_init", None),
        prompt_template=config["prompt_template"],
        aspect=config["aspect"],
        device=device
    ).to(device)

    ckpt_path = config["stage2a_ckpt"]
    prompt_learner.load_state_dict(torch.load(ckpt_path, map_location=device,weights_only=False))
    prompt_learner.eval()

    print(f"🧠 Loaded prompt weights from: {ckpt_path}")

    all_img_feats, all_txt_feats, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="🔎 Validating"):
        images, labels = images.to(device), labels.to(device)

        # image features
        img_feats = clip_model.encode_image(images)
        img_feats = F.normalize(img_feats, dim=-1)

        # text features
        prompts = prompt_learner.forward_batch(labels)
        pos_embed = clip_model.positional_embedding
        x = prompts + pos_embed.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x[:, 0, :])
        txt_feats = x @ clip_model.text_projection
        txt_feats = F.normalize(txt_feats, dim=-1)

        all_img_feats.append(img_feats)
        all_txt_feats.append(txt_feats)
        all_labels.append(labels)

    img_feats = torch.cat(all_img_feats)
    txt_feats = torch.cat(all_txt_feats)
    labels = torch.cat(all_labels)

    # Cosine similarity matrix
    sims = img_feats @ txt_feats.T
    top1 = (sims.argmax(dim=1) == torch.arange(len(labels), device=device)).float().mean().item()
    top5 = (sims.topk(5, dim=1).indices == torch.arange(len(labels), device=device).unsqueeze(1)).any(dim=1).float().mean().item()

    print("\n🎯 Validation Summary")
    print(f"Top-1 Accuracy : {top1:.4f}")
    print(f"Top-5 Accuracy : {top5:.4f}")


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate saved prompt learner from Stage 2a")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    validate(config)
