import os
import sys
import torch
import clip
from datetime import datetime
from datasets.build_dataloader import get_dataloader
from engine.baseline_inference import extract_features, compute_similarity_matrix
from engine.evaluator import evaluate_rank


def run_baseline_evaluation(log_file_path=None):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()

    # Output log buffer
    output_lines = []
    output_lines.append(f"📋 Starting CLIP ViT-B/16 baseline evaluation on 11k dataset")
    output_lines.append(f"🖥️ Device: {device}")

    # Set root path to 11k dataset (dorsal_r split assumed)
    root_path = "../datasets/11khands/train_val_test_split_dorsal_r"
    all_metrics = []

    for i in range(1):
        output_lines.append("\n" + "=" * 60)
        output_lines.append(f"🔁 Starting Evaluation: [Run {i+1}/10]")
        output_lines.append("=" * 60)

        query_dir = os.path.join(root_path, f"query{i}")
        gallery_dir = os.path.join(root_path, f"gallery{i}")

        # Load and log QUERY
        query_loader = get_dataloader(query_dir, batch_size=32, shuffle=False, train=False)
        num_query_images = len(query_loader.dataset)
        output_lines.append(f"\n📥 Extracting features from QUERY set (query{i})...")
        output_lines.append(f"🖼️ Total QUERY images: {num_query_images}")
        print("\n".join(output_lines[-2:]))  # Print progress

        query_features, query_labels = extract_features(model, query_loader, device)

        # Load and log GALLERY
        gallery_loader = get_dataloader(gallery_dir, batch_size=32, shuffle=False, train=False)
        num_gallery_images = len(gallery_loader.dataset)
        output_lines.append(f"\n📦 Extracting features from GALLERY set (gallery{i})...")
        output_lines.append(f"🖼️ Total GALLERY images: {num_gallery_images}")
        print("\n".join(output_lines[-2:]))

        gallery_features, gallery_labels = extract_features(model, gallery_loader, device)

        # Evaluate similarity
        output_lines.append(f"\n🔍 Evaluating similarity and metrics for Run {i+1}...")
        print(output_lines[-1])
        sim_matrix = compute_similarity_matrix(query_features, gallery_features)
        metrics = evaluate_rank(sim_matrix, query_labels, gallery_labels)

        output_lines.append(f"✅ Run-{i+1} Results:")
        for k, v in metrics.items():
            output_lines.append(f"   {k}: {v:.4f}")
        all_metrics.append(metrics)

    # Final averaged results
    avg_results = {}
    for key in all_metrics[0].keys():
        avg_results[key] = sum(d[key] for d in all_metrics) / len(all_metrics)

    output_lines.append("\n" + "=" * 60)
    output_lines.append("📊 Final Averaged Results (ViT-B/16 over 10 runs)")
    output_lines.append("=" * 60)
    for k, v in avg_results.items():
        output_lines.append(f"{k}: {v:.4f}")
    output_lines.append("=" * 60)

    # Print to console
    for line in output_lines:
        print(line)

    # Save to result log
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\n📁 Output also saved to: {log_file_path}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"../result_logs/baseline_vitb16_11k_eval_{timestamp}.log"
    run_baseline_evaluation(log_file_path=log_file_path)
