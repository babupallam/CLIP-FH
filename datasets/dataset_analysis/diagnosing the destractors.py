import os

def analyze_gallery_folder(gallery_path):
    folder_ids = []
    real_ids = []
    distractor_ids = []

    for folder in os.listdir(gallery_path):
        if not os.path.isdir(os.path.join(gallery_path, folder)):
            continue  # Skip files, only check folders
        try:
            folder_id = int(folder)
            folder_ids.append(folder_id)
            if folder_id < 1000:
                real_ids.append(folder_id)
            else:
                distractor_ids.append(folder_id)
        except ValueError:
            print(f"⚠️ Warning: Non-numeric folder name found → {folder}")

    print("\n✅ Gallery Folder Analysis")
    print("-" * 40)
    print(f"Total folders      : {len(folder_ids)}")
    print(f"Real identities    : {len(real_ids)} (IDs < 1000)")
    print(f"Distractor IDs     : {len(distractor_ids)} (IDs ≥ 1000)")

    overlap = set(real_ids) & set(distractor_ids)
    if overlap:
        print(f"\n❌ Collision detected! Same ID in both sets: {sorted(overlap)}")
    else:
        print("\n✅ No ID collisions detected.")

    print("-" * 40)

# Example usage
gallery_folder = "datasets/HD/Original Images/train_val_test_split/gallery0"

analyze_gallery_folder(gallery_folder)



def list_distractor_ids(gallery_path):
    distractors = []
    for folder in os.listdir(gallery_path):
        try:
            folder_id = int(folder)
            if folder_id >= 1000:
                distractors.append(folder_id)
        except:
            continue

    print(f"Found {len(distractors)} distractor ID folders:")
    print(sorted(distractors))

# Example usage
list_distractor_ids(gallery_folder)
