import sys, os, argparse, yaml, torch
# Setup project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if "datasets" not in os.listdir(PROJECT_ROOT):
    raise RuntimeError("PROJECT_ROOT is misaligned.")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from utils.dataloaders import get_train_val_loaders
from utils.train_helpers import build_model
from utils.naming import build_filename
from engine.train_classifier_stage1 import FinetuneTrainerStage1


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["experiment"] = config.get("experiment", "stage1")
    base_name = build_filename(config, config["epochs"], stage="image", extension="", timestamped=False)
    config["save_path"] = os.path.join(config["save_dir"], base_name + ".pth")
    config["log_path"] = os.path.join(config["log_dir"], base_name + ".log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    clip_model, classifier = build_model(config, freeze_text=True)
    trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, val_loader, config, device)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    main(args.config)
