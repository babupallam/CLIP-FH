import os

def run_all_eval_configs():
    # Absolute path to evaluation config directory
    config_dir = os.path.abspath("configs/eval_stage1_frozen_text")

    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"❌ Config directory not found: {config_dir}")

    # Only .yml files (ignore folders or other files)
    config_files = sorted([
        f for f in os.listdir(config_dir)
        if os.path.isfile(os.path.join(config_dir, f)) and f.endswith(".yml")
    ])

    # Get path to run_eval_clip.py
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    run_eval_path = os.path.join(root_dir, "experiments", "run_eval_clip.py")

    if not os.path.exists(run_eval_path):
        raise FileNotFoundError(f"❌ Could not find run_eval_clip.py at: {run_eval_path}")

    print("📂 Running all evaluation configs in:", config_dir)

    # Run each config using os.system
    for config in config_files:
        full_path = os.path.join(config_dir, config)
        print(f"\n🚀 Running Evaluation: {config}")
        os.system(f"python \"{run_eval_path}\" --config \"{full_path}\"")

if __name__ == "__main__":
    run_all_eval_configs()
