import os

def run_all_configs(config_dir):
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".yml")]
    for config in config_files:
        full_path = os.path.join(config_dir, config)
        print(f"\n🚀 Running: {config}")
        os.system(f"python run_eval_clip.py --config {full_path}")

if __name__ == "__main__":
    print("📂 Running all baseline evaluations...")
    run_all_configs("../configs/baseline/")

    # as make progress new models can be evaluated in this way
    # print("\n📂 Running all fine-tuned evaluations...")
    # run_all_configs("configs/finetuned/")
