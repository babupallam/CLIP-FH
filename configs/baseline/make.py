import os
import yaml

# Base folder to save YAML configs
config_dir = "configs/baseline"
os.makedirs(config_dir, exist_ok=True)

# Supported values
models = ["vitb16", "rn50"]
aspects_11k = ["dorsal_r", "dorsal_l", "palmar_r", "palmar_l"]
aspects_hd = ["hd"]

# Common config values
batch_size = 32
num_splits = 10

def write_config(config_data, filename):
    with open(filename, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)

# Generate configs for 11k
for model in models:
    for aspect in aspects_11k:
        cfg = {
            "dataset": "11k",
            "aspect": aspect,
            "model": model,
            "variant": "baseline",
            "batch_size": batch_size,
            "num_splits": num_splits
        }
        fname = f"eval_{model}_11k_{aspect}.yml"
        write_config(cfg, os.path.join(config_dir, fname))

# Generate configs for HD (only dorsal_r)
for model in models:
    for aspect in aspects_hd:
        cfg = {
            "dataset": "hd",
            "aspect": aspect,
            "model": model,
            "variant": "baseline",
            "batch_size": batch_size,
            "num_splits": num_splits
        }
        fname = f"eval_{model}_hd_{aspect}.yml"
        write_config(cfg, os.path.join(config_dir, fname))

print(f"✅ Generated {len(models)*(len(aspects_11k)+len(aspects_hd))} baseline configs in {config_dir}")
