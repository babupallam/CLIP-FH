import os
import sys
import argparse
import subprocess

def run_single_config(config_path):
    # Resolve full absolute path
    config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f" Config file not found: {config_path}")

    #  Locate run_eval_clip.py relative to this script
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    run_eval_path = os.path.join(root_dir, "experiments", "run_eval_clip.py")

    if not os.path.exists(run_eval_path):
        raise FileNotFoundError(f" Could not find run_eval_clip.py at: {run_eval_path}")

    print(f" Running Evaluation with config: {config_path}")
    os.system(f"python \"{run_eval_path}\" --config \"{config_path}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single baseline evaluation config.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    run_single_config(args.config)
