import os


def run_all_configs(config_dir):
    config_dir = os.path.abspath(config_dir)

    if not os.path.exists(config_dir):
        raise FileNotFoundError(f"❌ Config directory not found: {config_dir}")

    # 🔥 Only include .yml files directly inside baseline (ignore subfolders like 'rest')
    config_files = [f for f in os.listdir(config_dir)
                    if os.path.isfile(os.path.join(config_dir, f)) and f.endswith(".yml")]

    # 📍 Locate run_eval_clip.py relative to this script
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    run_eval_path = os.path.join(root_dir, "experiments", "run_eval_clip.py")

    if not os.path.exists(run_eval_path):
        raise FileNotFoundError(f"❌ Could not find run_eval_clip.py at: {run_eval_path}")

    for config in config_files:
        full_path = os.path.join(config_dir, config)
        print(f"\n🚀 Running: {config}")
        os.system(f"python \"{run_eval_path}\" --config \"{full_path}\"")


if __name__ == "__main__":
    print("📂 Running all baseline evaluations...")

    # ✅ Target only: configs/baseline/
    config_dir = os.path.join(os.path.dirname(__file__), "../../configs/baseline")
    run_all_configs(config_dir)
