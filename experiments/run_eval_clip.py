"""
run_evaluation.py

Purpose:
- Loads CLIP (ViT-B/16 or RN50), optionally loads fine-tuned weights.
- Loads query and gallery dataloaders for multiple splits of a dataset (11k or HD).
- Performs inference using CLIP image encoder to extract features.
- Computes similarity matrix and evaluates ReID performance using rank-based metrics.
- Logs and saves results for all splits.

Useful For:
- Comparing baseline vs. fine-tuned CLIP models
- Evaluating different model-dataset-aspect combinations
- Logging repeatable evaluation experiments
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ========= Imports =========

import os                          # For file path construction and checking
import torch                       # PyTorch: tensor operations and model loading
import yaml                        # For reading YAML config files
import argparse                    # For command-line argument parsing
from datetime import datetime      # For timestamping log file names

# Internal project imports for dataloader, inference, and evaluation
from datasets.build_dataloader import get_dataloader         # Loads image datasets into DataLoaders
from engine.baseline_inference import extract_features, compute_similarity_matrix  # Feature extraction and similarity
from engine.evaluator import evaluate_rank                   # Evaluation using CMC and mAP

# Load OpenAI CLIP (supports ViT-B/16 and RN50 variants)
import clip


def run_evaluation(config):
    """
    Main function to evaluate a CLIP model (baseline or fine-tuned) on query-gallery splits.

    Args:
        config (dict): YAML configuration with keys:
            - dataset : str  → "11k" or "hd"
            - aspect  : str  → e.g., "dorsal", "palmar"
            - model   : str  → "vitb16" or "rn50"
            - variant : str  → "baseline" or "finetuned"
            - batch_size : int (optional)
            - num_splits : int (optional)
    """

    # ==== Parse config values ====
    dataset = config["dataset"]               # Dataset name (e.g., "11k" or "hd")
    aspect = config["aspect"]                 # Hand aspect (e.g., "dorsal", "palmar")
    model_name = config["model"]              # Model variant (e.g., "vitb16")
    variant = config["variant"]               # "baseline" or "finetuned"
    batch_size = config.get("batch_size", 32) # Default to 32 if not given
    num_splits = config.get("num_splits", 10) # Number of cross-validation splits

    # ==== Set device ====
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU if available

    # ==== Load CLIP model ====
    model_id = "ViT-B/16" if model_name == "vitb16" else "RN50"  # Select model architecture
    model, _ = clip.load(model_id, device=device)                # Load pretrained CLIP
    model.eval()                                                 # Set to eval mode (no dropout, no grad)

    # ==== Load fine-tuned weights (if applicable) ====
    if variant == "finetuned":
        ckpt_name = f"finetuned_{model_name}_{dataset}_{aspect}.pth"
        ckpt_path = os.path.join("saved_models", ckpt_name)

        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))  # Load weights
            print(f"✅ Loaded fine-tuned weights from: {ckpt_path}")
        else:
            print(f"❌ Fine-tuned model not found: {ckpt_path}")
            return  # Exit if checkpoint missing

    # ==== Define dataset path ====
    if dataset == "11k":
        data_path = os.path.join("datasets", "11khands", f"train_val_test_split_{aspect}")
    elif dataset == "hd":
        data_path = os.path.join("datasets", "HD", "Original Images", "train_val_test_split")
    else:
        raise ValueError("Unsupported dataset name in config.")

    # ==== Setup logging ====
    all_metrics = []        # Stores metrics from each split
    output_lines = []       # Stores text to be logged
    output_lines.append(f"📋 CONFIG: {model_name.upper()} | {dataset.upper()} | {aspect} | {variant.upper()}")

    # ==== Run evaluation over all splits ====
    for i in range(num_splits):
        # Set paths for query and gallery for the ith split
        query_path = os.path.join(data_path, f"query{i}")
        gallery_path = os.path.join(data_path, f"gallery{i}")

        # 🧪 Debug check
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"❌ Query path does not exist: {query_path}")
        if not os.path.exists(gallery_path):
            raise FileNotFoundError(f"❌ Gallery path does not exist: {gallery_path}")

        # Load DataLoaders for query and gallery sets
        query_loader = get_dataloader(query_path, batch_size=batch_size, shuffle=False, train=False)
        gallery_loader = get_dataloader(gallery_path, batch_size=batch_size, shuffle=False, train=False)

        # Logging run header
        run_header = (
            f"\n{'=' * 60}\n"
            f"🔁 Run {i + 1}/{num_splits} | 🖼️ Query: query{i} ({len(query_loader.dataset)} images) "
            f"| Gallery: gallery{i} ({len(gallery_loader.dataset)} images)"
        )
        output_lines.append(run_header)
        print(run_header)

        # Extract visual features using CLIP image encoder
        query_feats, query_labels = extract_features(model, query_loader, device)
        gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

        # Compute cosine similarity matrix between query and gallery
        sim_matrix = compute_similarity_matrix(query_feats, gallery_feats)

        # Evaluate using CMC, mAP, and rank-k accuracy
        metrics = evaluate_rank(sim_matrix, query_labels, gallery_labels)

        # Log results for this split
        output_lines.append(f"✅ Run-{i+1} Results:")
        for k, v in metrics.items():
            output_lines.append(f"   {k}: {v:.4f}")
        all_metrics.append(metrics)

    # ==== Average metrics across all splits ====
    avg_results = {k: sum(m[k] for m in all_metrics)/num_splits for k in all_metrics[0]}
    output_lines.append(f"\n{'='*60}")
    output_lines.append(f"📊 FINAL AVERAGED RESULTS ({model_name.upper()} - {dataset.upper()} - {aspect})")
    output_lines.append("=" * 60)
    for k, v in avg_results.items():
        output_lines.append(f"{k}: {v:.4f}")
    output_lines.append("=" * 60)

    # ==== Print all results to terminal ====
    for line in output_lines:
        print(line)

    # ==== Save results to log file ====
    os.makedirs("result_logs", exist_ok=True)  # Make log directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique filename
    log_filename = f"{variant}_{model_name}_{dataset}_{aspect}_{timestamp}.log"
    log_path = os.path.join("../result_logs", log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n📁 Results saved to: {log_path}")


# ==== Entry point ====
if __name__ == "__main__":
    # Parse command-line argument to load config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config file as dictionary
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Start evaluation
    run_evaluation(config)
