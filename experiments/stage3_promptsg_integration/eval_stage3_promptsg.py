import os
import sys

def run_eval(config_path):
    """
    Launches ReID evaluation for Stage 3 PromptSG (query-gallery splits).
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Path to main evaluation runner
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    eval_runner = os.path.join(root_dir, "experiments", "run_eval_clip.py")

    if not os.path.exists(eval_runner):
        raise FileNotFoundError(f"Missing: {eval_runner}")

    print(f"\nLaunching Stage 3 Evaluation for PromptSG")
    print(f"Using config: {config_path}")
    os.system(f"python \"{eval_runner}\" --config \"{config_path}\"")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_stage3_promptsg.py <path_to_eval_config.yml>")
        sys.exit(1)

    run_eval(sys.argv[1])
