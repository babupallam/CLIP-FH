import os
import re
import pandas as pd

# Directory containing all train logs for stage 3
TRAIN_LOG_DIR = 'train_logs'
OUTPUT_DIR = 'result_logs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to parse a single log file
def parse_stage3_promptsg_log(filepath, version, model_name):
    data = []
    current_epoch = None
    train_loss = None
    loss_id = loss_triplet = loss_center = loss_i2t = loss_t2i = None
    val_rank1 = val_rank5 = val_rank10 = val_map = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Epoch" in line and "Validating" not in line:
            match = re.search(r'Epoch (\d+)', line)
            if match:
                current_epoch = int(match.group(1))

        if '[Epoch' in line and 'Total Loss=' in line:
            match = re.search(r'Total Loss=([0-9.]+)', line)
            if match:
                train_loss = float(match.group(1))

        if "Loss Breakdown" in line:
            match = re.findall(r"(\w+)\s*=\s*([0-9.]+)", line)
            for key, val in match:
                if key == "ID":
                    loss_id = float(val)
                elif key == "Triplet":
                    loss_triplet = float(val)
                elif key == "Center":
                    loss_center = float(val)
                elif key == "i2t":
                    loss_i2t = float(val)
                elif key == "t2i":
                    loss_t2i = float(val)

        if 'Validation rank-1' in line:
            val_rank1 = float(re.search(r'(\d+\.\d+)', line).group(1))
        elif 'Validation rank-5' in line:
            val_rank5 = float(re.search(r'(\d+\.\d+)', line).group(1))
        elif 'Validation rank-10' in line:
            val_rank10 = float(re.search(r'(\d+\.\d+)', line).group(1))
        elif 'Validation mAP' in line:
            val_map = float(re.search(r'(\d+\.\d+)', line).group(1))
            is_best = "yes" if i + 1 < len(lines) and "New BEST" in lines[i + 1] else "no"

            if current_epoch is not None:
                data.append({
                    'version': version,
                    'epoch': current_epoch,
                    'train_loss': train_loss,
                    'loss_id': loss_id,
                    'loss_triplet': loss_triplet,
                    'loss_center': loss_center,
                    'loss_i2t': loss_i2t,
                    'loss_t2i': loss_t2i,
                    'val_rank1': val_rank1,
                    'val_rank5': val_rank5,
                    'val_rank10': val_rank10,
                    'val_mAP': val_map,
                    'is_best': is_best,
                    'model': model_name
                })

    return data

# Function to gather all logs across versions for a given model
def collect_logs_stage3(model_name, n_versions=11):
    all_data = []
    for v in range(1, n_versions + 1):
        version_name = f"stage3-v{v}"
        subdir = os.path.join(TRAIN_LOG_DIR, version_name)
        if not os.path.exists(subdir):
            continue
        found = False
        for fname in os.listdir(subdir):
            if model_name in fname.lower() and fname.endswith(".log"):
                log_path = os.path.join(subdir, fname)
                print(f"[] Parsing {log_path}")
                parsed = parse_stage3_promptsg_log(log_path, version_name, model_name)
                all_data.extend(parsed)
                found = True
                break
        if not found:
            print(f"[!] Skipping {model_name} in {version_name} (log file not found)")
    return pd.DataFrame(all_data)

# Run the parser for both architectures
if __name__ == "__main__":
    df_vitb = collect_logs_stage3("vitb16")
    df_rn50 = collect_logs_stage3("rn50")

    if not df_vitb.empty:
        df_vitb.to_csv(os.path.join(OUTPUT_DIR, "stage3_vitb16_train_table.csv"), index=False)
    if not df_rn50.empty:
        df_rn50.to_csv(os.path.join(OUTPUT_DIR, "stage3_rn50_train_table.csv"), index=False)

    print("Stage 3 training metrics saved to result_logs/")
