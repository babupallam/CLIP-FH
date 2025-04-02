import os
import sys

# 🔧 Ensure local module imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if "datasets" not in os.listdir(PROJECT_ROOT):
    raise RuntimeError("❌ PROJECT_ROOT is misaligned. Check relative path in script.")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===== External Libraries =====
import yaml
import argparse
import torch
from datetime import datetime

# ===== Internal Modules =====
from models.prompt_learner import PromptLearner
from models.clip_patch import load_clip_with_patch
from engine.clipreid_trainer_stage1 import PromptLearnerTrainerStage1
from datasets.build_dataloader import get_train_val_loaders


def main(config_path):
    # 🔹 Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 🔹 Extract core identifiers
    model_type = config["model"]
    dataset = config["dataset"]
    aspect = config["aspect"]
    exp_name = config["experiment"]

    # 🔹 Extract training hyperparameters
    n_ctx = config.get("n_ctx", 8)
    lr = config.get("lr", 0.0001)
    bs = config.get("batch_size", 32)
    epochs = config.get("epochs", 20)
    ctx_init_raw = config.get("ctx_init")
    ctx_init = str(ctx_init_raw).replace(" ", "_") if ctx_init_raw is not None else "none"

    # 🔹 Auto-generate filename
    suffix = f"nctx{n_ctx}_e{epochs}_lr{str(lr).replace('.', '')}_bs{bs}_ctx{ctx_init}"
    base_filename = f"stage2a_prompt_{model_type}_{dataset}_{aspect}_{suffix}"

    # 🔹 Timestamped log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 🔹 Prepare full paths
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    config["save_path"] = os.path.join(config["save_dir"], base_filename + ".pth")
    config["log_path"] = os.path.join(config["output_dir"], base_filename + f"_{timestamp}.log")

    # 🔹 Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")

    # 🔹 Load frozen CLIP
    clip_model, preprocess = load_clip_with_patch(model_type, device, freeze_all=True)

    # 🔹 Load training data
    train_loader, _, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    # 🔄 Align classnames to ImageFolder's internal label ordering
    class_to_idx = train_loader.dataset.class_to_idx
    classnames = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    print(f"classnames{classnames}")

    # 🔹 Initialize prompt learner
    prompt_learner = PromptLearner(
        classnames=classnames,
        clip_model=clip_model,
        n_ctx=n_ctx,
        ctx_init=config.get("ctx_init", None),
        prompt_template=config["prompt_template"],
        device=device
    )
    #print("Initialised prompt learner")

    # 🔹 Train prompt learner
    trainer = PromptLearnerTrainerStage1(
        clip_model=clip_model,
        prompt_learner=prompt_learner,
        train_loader=train_loader,
        config=config,
        device=device
    )

    #trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
