import os
import torch
import yaml
import argparse
from datetime import datetime

from datasets.build_dataloader import get_dataloader
from engine.baseline_inference import extract_features, compute_similarity_matrix
from engine.evaluator import evaluate_rank


def run_evaluation(config):
    dataset = config["dataset"]
    aspect = config["aspect"]
    model_name = config["model"]
    variant = config["variant"]
    batch_size = config.get("batch_size", 32)
    num_splits = config.get("num_splits", 10)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    import clip
    model_id = "ViT-B/16" if model_name == "vitb16" else "RN50"
    model, _ = clip.load(model_id, device=device)
    model.eval()

    if variant == "finetuned":
        ckpt_name = f"finetuned_{model_name}_{dataset}_{aspect}.pth"
        ckpt_path = os.path.join("saved_models", ckpt_name)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"✅ Loaded fine-tuned weights from: {ckpt_path}")
        else:
            print(f"❌ Fine-tuned model not found: {ckpt_path}")
            return

    # Dataset path
    if dataset == "11k":
        data_path = f"../datasets/11khands/train_val_test_split_{aspect}"
    elif dataset == "hd":
        data_path = "../datasets/HD/Original Images/train_val_test_split"
    else:
        raise ValueError("Unsupported dataset name in config.")

    all_metrics = []
    output_lines = []
    output_lines.append(f"📋 CONFIG: {model_name.upper()} | {dataset.upper()} | {aspect} | {variant.upper()}")

    for i in range(num_splits):
        query_path = os.path.join(data_path, f"query{i}")
        gallery_path = os.path.join(data_path, f"gallery{i}")

        query_loader = get_dataloader(query_path, batch_size=batch_size, shuffle=False, train=False)
        gallery_loader = get_dataloader(gallery_path, batch_size=batch_size, shuffle=False, train=False)

        # Build and print progress string
        run_header = (
            f"\n{'=' * 60}\n"
            f"🔁 Run {i + 1}/{num_splits} | 🖼️ Query: query{i} ({len(query_loader.dataset)} images) "
            f"| Gallery: gallery{i} ({len(gallery_loader.dataset)} images)"
        )

        # Append to logs and print to terminal
        output_lines.append(run_header)
        print(run_header)

        query_feats, query_labels = extract_features(model, query_loader, device)
        gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)

        sim_matrix = compute_similarity_matrix(query_feats, gallery_feats)
        metrics = evaluate_rank(sim_matrix, query_labels, gallery_labels)

        output_lines.append(f"✅ Run-{i+1} Results:")
        for k, v in metrics.items():
            output_lines.append(f"   {k}: {v:.4f}")
        all_metrics.append(metrics)

    # Average Results
    avg_results = {k: sum(m[k] for m in all_metrics)/num_splits for k in all_metrics[0]}
    output_lines.append(f"\n{'='*60}")
    output_lines.append(f"📊 FINAL AVERAGED RESULTS ({model_name.upper()} - {dataset.upper()} - {aspect})")
    output_lines.append("=" * 60)
    for k, v in avg_results.items():
        output_lines.append(f"{k}: {v:.4f}")
    output_lines.append("=" * 60)

    # Show in terminal
    for line in output_lines:
        print(line)

    # Save to log file
    os.makedirs("result_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{variant}_{model_name}_{dataset}_{aspect}_{timestamp}.log"
    log_path = os.path.join("../result_logs", log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n📁 Results saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_evaluation(config)
