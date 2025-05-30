import os
import re
import pandas as pd

# Regex patterns to extract different parts of the log
patterns = {
    "epoch": re.compile(r"\[Epoch (\d+)]"),
    "prompt_loss": re.compile(r"Avg Prompt Loss: ([\d.]+)"),
    "learning_rate": re.compile(r"Learning Rate: ([\d.eE+-]+)"),
    "loss_id": re.compile(r"ID = ([\d.]+)"),
    "loss_triplet": re.compile(r"Triplet = ([\d.]+)"),
    "loss_center": re.compile(r"Center = ([\d.]+)"),
    "loss_i2t": re.compile(r"i2t = ([\d.]+)"),
    "loss_t2i": re.compile(r"t2i = ([\d.]+)"),
    "rank1": re.compile(r"RANK1: ([\d.]+)%"),
    "rank5": re.compile(r"RANK5: ([\d.]+)%"),
    "rank10": re.compile(r"RANK10: ([\d.]+)%"),
    "map": re.compile(r"MAP: ([\d.]+)%"),
    "best_flag": re.compile(r"New BEST", re.IGNORECASE)
}

# Parse a single .log file
def parse_log(filepath, version):
    rows = []
    current = {"version": version}
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        for key, pattern in patterns.items():
            match = pattern.search(line)
            if match and match.lastindex:
                value = float(match.group(1)) if key != "epoch" else int(match.group(1))
                current[key] = value

        # If we've just completed a validation block (mAP is the last to show up)
        if "map" in current:
            current["is_best"] = "yes" if i+1 < len(lines) and "New BEST" in lines[i+1] else "no"
            rows.append(current.copy())
            # Reset for next epoch but keep version
            current = {"version": version}

    return pd.DataFrame(rows)

# Walk through all stage2-v* folders and gather per-arch logs
def collect_all_logs(root="train_logs", arch="vitb16"):
    all_data = []
    for i in range(1, 11):
        version = f"stage2-v{i}"
        folder = os.path.join(root, version)
        if not os.path.isdir(folder):
            continue
        found = False
        for fname in os.listdir(folder):
            if arch in fname.lower() and fname.endswith(".log"):
                path = os.path.join(folder, fname)
                df = parse_log(path, version)
                all_data.append(df)
                found = True
                break
        if not found:
            print(f"[INFO] No log file found for {arch} in {version}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Run and save both CSVs
if __name__ == "__main__":
    os.makedirs("result_logs", exist_ok=True)

    vit_df = collect_all_logs("train_logs", arch="vitb16")
    rn_df = collect_all_logs("train_logs", arch="rn50")

    vit_df.to_csv("result_logs/stage2_vitb16_train_table.csv", index=False)
    rn_df.to_csv("result_logs/stage2_rn50_train_table.csv", index=False)

    print("CSVs with all epoch results saved to result_logs/")
