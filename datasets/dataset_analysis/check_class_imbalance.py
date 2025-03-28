import os
from collections import Counter

# Create log file
os.makedirs("result_logs", exist_ok=True)
log_file = "result_logs/dataset_class_imbalance_check.txt"
open(log_file, "w").close()

# 11k dataset paths and structure
aspects_11k = [
    "train_val_test_split_dorsal_r",
    "train_val_test_split_dorsal_l",
    "train_val_test_split_palmar_r",
    "train_val_test_split_palmar_l"
]
splits = ["train", "val", "test"]
monte_prefixes = [f"gallery{i}" for i in range(10)] + [f"query{i}" for i in range(10)]

# HD dataset path
hd_root = "./datasets/HD/Original Images/train_val_test_split"
root_11k = "./datasets/11khands"

def print_and_log(line):
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def analyze_class_distribution(path):
    """
    For a given split folder (e.g., train), collect how many samples each class has.
    """
    sample_counts = []
    if not os.path.exists(path):
        return []

    for cls in os.listdir(path):
        class_path = os.path.join(path, cls)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            sample_counts.append(count)

    return sample_counts

def check_imbalance(dataset_name, sample_counts):
    """
    Given sample counts per class, check for imbalance and report range + warning if needed.
    """
    if not sample_counts:
        print_and_log(f"{dataset_name:<30} | ❌ Not Found or Empty")
        return

    min_val = min(sample_counts)
    max_val = max(sample_counts)
    balanced = min_val == max_val
    classes = len(sample_counts)

    msg = f"{dataset_name:<30} | Classes: {classes:>4} | Min: {min_val:>2} | Max: {max_val:>2} | "
    msg += "✅ Balanced" if balanced else "⚠️ Imbalanced"
    print_and_log(msg)

# 🧪 Step 1: Analyze 11k - train/val/test
print_and_log("📊 Class Imbalance Check: 11k Dataset")
print_and_log("=" * 70)

for aspect in aspects_11k:
    aspect_name = aspect.replace("train_val_test_split_", "")
    for split in splits:
        split_path = os.path.join(root_11k, aspect, split)
        sample_counts = analyze_class_distribution(split_path)
        check_imbalance(f"{aspect_name}_{split}", sample_counts)

# 🧪 Step 2: Analyze 11k - Monte Carlo folders
print_and_log("\n🎲 Monte Carlo Splits: 11k Dataset")
print_and_log("=" * 70)

for aspect in aspects_11k:
    aspect_name = aspect.replace("train_val_test_split_", "")
    aspect_path = os.path.join(root_11k, aspect)
    for prefix in monte_prefixes:
        split_path = os.path.join(aspect_path, prefix)
        sample_counts = analyze_class_distribution(split_path)
        check_imbalance(f"{aspect_name}_{prefix}", sample_counts)

# 🧪 Step 3: Analyze HD - train/val/test
print_and_log("\n📊 Class Imbalance Check: HD Dataset")
print_and_log("=" * 70)

for split in splits:
    split_path = os.path.join(hd_root, split)
    sample_counts = analyze_class_distribution(split_path)
    check_imbalance(f"hd_{split}", sample_counts)

# 🧪 Step 4: Analyze HD - Monte Carlo folders
print_and_log("\n🎲 Monte Carlo Splits: HD Dataset")
print_and_log("=" * 70)

for prefix in monte_prefixes:
    split_path = os.path.join(hd_root, prefix)
    sample_counts = analyze_class_distribution(split_path)
    check_imbalance(f"hd_{prefix}", sample_counts)

print_and_log(f"\n✅ Log saved to: {log_file}")
