import os
import subprocess

CONFIG_DIR = "configs/eval_stage1_frozen_text"

def run_all_eval_configs():
    config_files = sorted([f for f in os.listdir(CONFIG_DIR) if f.endswith(".yml")])

    for cfg in config_files:
        full_path = os.path.join(CONFIG_DIR, cfg)
        print(f"\n🚀 Running Evaluation: {cfg}")
        subprocess.run([
            "python", "experiments/run_eval_clip.py",
            "--config", full_path
        ])

if __name__ == "__main__":
    run_all_eval_configs()
