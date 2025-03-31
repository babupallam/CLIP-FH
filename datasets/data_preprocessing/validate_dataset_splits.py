import os
from collections import defaultdict

def validate_id_dirs(base_path, expected_subdirs):
    """Check if all required subdirectories exist."""
    print(f"\n🔍 Checking: {base_path}")
    for sub in expected_subdirs:
        full_path = os.path.join(base_path, sub)
        if not os.path.isdir(full_path):
            print(f"❌ MISSING: {sub}")
        else:
            print(f"✅ Exists: {sub}")

def check_id_distribution(split_path, name, min_expected=1):
    """Check that each ID has at least one image in the split."""
    missing = []
    for pid in os.listdir(split_path):
        person_dir = os.path.join(split_path, pid)
        if not os.path.isdir(person_dir):
            continue
        num_images = len([f for f in os.listdir(person_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
        if num_images < min_expected:
            missing.append((pid, num_images))

    if missing:
        print(f"⚠️  {name}: Found {len(missing)} IDs with fewer than {min_expected} images")
        for pid, count in missing[:5]:
            print(f"   - ID {pid} has {count} images")
    else:
        print(f"✅ {name}: All IDs have ≥ {min_expected} images")

def verify_no_overlap(set1_path, set2_path, name1, name2):
    """Ensure there are no overlapping IDs between two splits."""
    ids1 = set(os.listdir(set1_path))
    ids2 = set(os.listdir(set2_path))
    overlap = ids1 & ids2
    if overlap:
        print(f"❌ Overlap found between {name1} and {name2}: {len(overlap)} IDs")
        print(f"   Sample overlapping ID: {list(overlap)[0]}")
    else:
        print(f"✅ No ID overlap between {name1} and {name2}")

def validate_query_gallery_pairs(base_path, runs=10):
    """Validate that query and gallery folders exist and contain matching IDs."""
    for i in range(runs):
        query_path = os.path.join(base_path, f"query{i}")
        gallery_path = os.path.join(base_path, f"gallery{i}")

        if not os.path.isdir(query_path) or not os.path.isdir(gallery_path):
            print(f"❌ query{i} or gallery{i} missing.")
            continue

        query_ids = set(os.listdir(query_path))
        gallery_ids = set(os.listdir(gallery_path))

        if not query_ids:
            print(f"❌ query{i} is empty.")
        if not gallery_ids:
            print(f"❌ gallery{i} is empty.")

        # Check if query IDs are present in gallery too
        missing = query_ids - gallery_ids
        if missing:
            print(f"⚠️  query{i}: {len(missing)} IDs not found in gallery{i}")
        else:
            print(f"✅ query{i} and gallery{i} contain aligned IDs")

def validate_hd_extra_subjects(gallery_path, threshold=1000):
    """Check that HD gallery includes added extra subjects."""
    gallery_ids = os.listdir(gallery_path)
    extra_ids = [pid for pid in gallery_ids if pid.isdigit() and int(pid) > threshold]
    print(f"✅ HD Gallery contains {len(extra_ids)} extra subjects (expected ≈ 211).")

def validate_11k_split(root_path):
    print("\n===========================\n🧪 VALIDATING 11K SPLIT\n===========================")
    expected_dirs = ['train_all', 'train', 'val', 'test'] + [f'query{i}' for i in range(10)] + [f'gallery{i}' for i in range(10)]
    validate_id_dirs(root_path, expected_dirs)

    check_id_distribution(os.path.join(root_path, 'train'), "Train")
    check_id_distribution(os.path.join(root_path, 'val'), "Val")
    check_id_distribution(os.path.join(root_path, 'test'), "Test")

    verify_no_overlap(os.path.join(root_path, 'train'), os.path.join(root_path, 'val'), "Train", "Val")
    verify_no_overlap(os.path.join(root_path, 'train'), os.path.join(root_path, 'test'), "Train", "Test")

    validate_query_gallery_pairs(root_path, runs=10)

def validate_hd_split(root_path):
    print("\n===========================\n🧪 VALIDATING HD SPLIT\n===========================")
    expected_dirs = ['train_all', 'train', 'val', 'test'] + [f'query{i}' for i in range(10)] + [f'gallery{i}' for i in range(10)]
    validate_id_dirs(root_path, expected_dirs)

    check_id_distribution(os.path.join(root_path, 'train'), "Train")
    check_id_distribution(os.path.join(root_path, 'val'), "Val")
    check_id_distribution(os.path.join(root_path, 'test'), "Test")

    verify_no_overlap(os.path.join(root_path, 'train'), os.path.join(root_path, 'val'), "Train", "Val")
    verify_no_overlap(os.path.join(root_path, 'train'), os.path.join(root_path, 'test'), "Train", "Test")

    validate_query_gallery_pairs(root_path, runs=10)

    # Check extra IDs in gallery
    validate_hd_extra_subjects(os.path.join(root_path, 'gallery0'))

if __name__ == "__main__":
    # Replace these paths with your actual dataset output locations
    path_11k = "./datasets/11khands/train_val_test_split_dorsal_r"
    path_hd = "./datasets/HD/Original Images/train_val_test_split"

    validate_11k_split(path_11k)
    validate_hd_split(path_hd)
