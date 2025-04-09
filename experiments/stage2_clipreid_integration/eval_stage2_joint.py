import os
import sys

def run_single_eval_config(config_path):
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Locate run_eval_clip.py
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    run_eval_path = os.path.join(root_dir, "experiments", "run_eval_clip.py")

    if not os.path.exists(run_eval_path):
        raise FileNotFoundError(f"Could not find run_eval_clip.py at: {run_eval_path}")

    print(f"\nRunning Evaluation (Stage 2 Joint) with config:\n📄 {config_path}\n")
    os.system(f"python \"{run_eval_path}\" --config \"{config_path}\"")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_stage2_joint.py <path_to_config.yml>")
        print("    e.g. python eval_stage2_joint.py configs/eval_stage2_joint/eval_vitb16_11k_dorsal_r.yml")
        sys.exit(1)

    config_file_path = sys.argv[1]
    run_single_eval_config(config_file_path)
