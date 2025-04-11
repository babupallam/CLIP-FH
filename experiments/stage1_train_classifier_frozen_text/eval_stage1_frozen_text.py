import os
import sys

def run_single_eval_config(config_path):
    """
    Runs a single evaluation config file for stage1_frozen_text evaluation.

    Args:
        config_path (str): Full or relative path to a YAML config file.
    """

    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    # Locate run_eval_clip.py
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    run_eval_path = os.path.join(root_dir, "experiments", "run_eval_clip.py")

    if not os.path.exists(run_eval_path):
        raise FileNotFoundError(f"❌ Could not find run_eval_clip.py at: {run_eval_path}")

    print(f"\n🚀 Running Evaluation with config: {config_path}")
    os.system(f"python \"{run_eval_path}\" --config \"{config_path}\"")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("⚠️ Usage: python eval_stage1_frozen_text.py <full_path_to_config.yml>")
        print("    e.g. python eval_stage1_frozen_text.py configs/eval_stage1_frozen_text/eval_vitb16_11k_dorsal_r.yml")
        sys.exit(1)

    config_file_path = sys.argv[1]
    run_single_eval_config(config_file_path)
