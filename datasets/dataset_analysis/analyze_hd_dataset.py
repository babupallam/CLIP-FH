import os
from collections import Counter

# Prepare logging
os.makedirs("result_logs", exist_ok=True)
log_file = "result_logs/dataset_hd_summary.txt"
open(log_file, "w").close()

# Dataset paths
hd_root = "./datasets/HD/Original Images/train_val_test_split"
splits = ["train", "val", "test"]
monte_prefixes = [f"gallery{i}" for i in range(10)] + [f"query{i}" for i in range(10)]

# Utilities
def print_and_log(line):
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def analyze_split(split_path):
    samples_per_class = []
    if not os.path.exists(split_path):
        return {
            "classes": 0, "samples": 0,
            "min": 0, "min_count": 0,
            "max": 0, "max_count": 0
        }

    for class_id in os.listdir(split_path):
        class_folder = os.path.join(split_path, class_id)
        if os.path.isdir(class_folder):
            count = len([
                f for f in os.listdir(class_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            samples_per_class.append(count)

    count_freq = Counter(samples_per_class)
    min_val = min(samples_per_class) if samples_per_class else 0
    max_val = max(samples_per_class) if samples_per_class else 0

    return {
        "classes": len(samples_per_class),
        "samples": sum(samples_per_class),
        "min": min_val,
        "min_count": count_freq.get(min_val, 0),
        "max": max_val,
        "max_count": count_freq.get(max_val, 0)
    }

def print_split_stats(split_label, stats):
    label = f"{split_label}".ljust(20)
    print_and_log(
        f"{label} | Classes: {stats['classes']:>4} | Samples: {stats['samples']:>5} "
        f"| Min: {stats['min']:>2} ({stats['min_count']}) | Max: {stats['max']:>2} ({stats['max_count']})"
    )

# 📘 Header: Explanation
print_and_log("📘 Column Explanation:")
print_and_log("-" * 85)
print_and_log("SPLIT LABEL         : Split type or Monte Carlo folder (e.g. TRAIN, gallery0)")
print_and_log("Classes             : Number of identity folders (unique IDs)")
print_and_log("Samples             : Total number of images in the split")
print_and_log("Min                 : Fewest images in a class + count of such classes")
print_and_log("Max                 : Most images in a class + count of such classes")
print_and_log("-" * 85)

# 📊 Summary for train/val/test
print_and_log("\n📊 HD Dataset Distribution Summary")
print_and_log("=" * 85)

for split in splits:
    split_path = os.path.join(hd_root, split)
    stats = analyze_split(split_path)
    print_split_stats(split.upper(), stats)

# 🎲 Monte Carlo analysis
print_and_log("\n🎲 Monte Carlo Query/Gallery Splits Summary")
print_and_log("=" * 85)

for prefix in monte_prefixes:
    split_path = os.path.join(hd_root, prefix)
    if os.path.exists(split_path):
        stats = analyze_split(split_path)
        print_split_stats(prefix, stats)

print_and_log(f"\n✅ Log saved to: {log_file}")
