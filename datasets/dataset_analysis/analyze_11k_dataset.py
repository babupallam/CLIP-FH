import os
from collections import Counter

# Log setup
os.makedirs("result_logs", exist_ok=True)
log_file = "result_logs/dataset_11k_summary.txt"
open(log_file, "w").close()

# Paths and structure
root_dir = "./datasets/11khands"
aspects = [
    "train_val_test_split_dorsal_r",
    "train_val_test_split_dorsal_l",
    "train_val_test_split_palmar_r",
    "train_val_test_split_palmar_l"
]
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

def print_split_stats(aspect, split, stats):
    label = f"{aspect}_{split}".ljust(20)
    print_and_log(
        f"{label} | Classes: {stats['classes']:>4} | Samples: {stats['samples']:>5} "
        f"| Min: {stats['min']:>2} ({stats['min_count']}) | Max: {stats['max']:>2} ({stats['max_count']})"
    )

# 📘 Header: Explanation
print_and_log("📘 Column Explanation:")
print_and_log("-" * 85)
print_and_log("ASPECT_SPLIT       : Aspect + split type (e.g. dorsal_r_train)")
print_and_log("Classes            : Number of identity folders (unique IDs)")
print_and_log("Samples            : Total number of images in the split")
print_and_log("Min                : Minimum samples found in any class (and how many classes have this count)")
print_and_log("Max                : Maximum samples found in any class (and how many classes have this count)")
print_and_log("-" * 85)

# 📊 Analysis of train/val/test
print_and_log("\n📊 11k Dataset Distribution Summary")
print_and_log("=" * 85)

for aspect_path in aspects:
    aspect_name = aspect_path.replace("train_val_test_split_", "")
    print_and_log(f"\n🔹 {aspect_name.upper()}")
    for split in splits:
        split_path = os.path.join(root_dir, aspect_path, split)
        stats = analyze_split(split_path)
        print_split_stats(aspect_name, split, stats)

# 🎲 Monte Carlo analysis
print_and_log("\n🎲 Monte Carlo Query/Gallery Splits Summary")
print_and_log("=" * 85)

for aspect_path in aspects:
    aspect_name = aspect_path.replace("train_val_test_split_", "")
    monte_root = os.path.join(root_dir, aspect_path)
    found = False
    for prefix in monte_prefixes:
        split_path = os.path.join(monte_root, prefix)
        if os.path.exists(split_path):
            if not found:
                print_and_log(f"\n🔹 {aspect_name.upper()}")
                found = True
            stats = analyze_split(split_path)
            print_split_stats(aspect_name, prefix, stats)

print_and_log(f"\n✅ Log saved to: {log_file}")
