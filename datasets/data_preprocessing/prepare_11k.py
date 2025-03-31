"""
prepare_11k.py

Data Preparation Script for 11k Hand Dataset.
- Splits data into train, val, test, query, and gallery folders.
- Supports dorsal and palmar aspects (right and left).
"""

import os
import csv
import argparse
import numpy as np
from shutil import copyfile

def create_dirs(base_path, aspects, splits, query_gallery_splits=10):
    """Creates directory structures for train/val/test/query/gallery."""
    for aspect in aspects:
        aspect_dir = os.path.join(base_path, f"train_val_test_split_{aspect}")
        os.makedirs(aspect_dir, exist_ok=True)

        # Create splits: train, val, test, train_all
        for split in splits:
            split_path = os.path.join(aspect_dir, split)
            os.makedirs(split_path, exist_ok=True)

        # Create query and gallery folders (query0, gallery0, ...)
        for i in range(query_gallery_splits):
            os.makedirs(os.path.join(aspect_dir, f"query{i}"), exist_ok=True)
            os.makedirs(os.path.join(aspect_dir, f"gallery{i}"), exist_ok=True)

def read_hand_info(hand_info_path):
    """Reads the HandInfo CSV."""
    with open(hand_info_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def split_data_by_aspect(data, train_path, save_base, aspect, id_threshold_train, id_threshold_test):
    """Splits data into train_all and test by aspect."""
    train_all_path = os.path.join(save_base, "train_all")
    test_path = os.path.join(save_base, "test")

    for row in data:
        if row['aspectOfHand'] != aspect or int(row['accessories']) != 0:
            continue

        person_id = int(row['id'])
        image_name = row['imageName']
        src_path = os.path.join(train_path, image_name)

        # Train or test assignment
        if person_id <= id_threshold_train:
            dst_dir = os.path.join(train_all_path, str(person_id))
        else:
            dst_dir = os.path.join(test_path, str(person_id))

        os.makedirs(dst_dir, exist_ok=True)
        copyfile(src_path, os.path.join(dst_dir, image_name))

def split_train_val(train_all_path, train_path, val_path, ext='jpg'):
    """Randomly splits train_all into train and val."""
    for person_id in os.listdir(train_all_path):
        person_dir = os.path.join(train_all_path, person_id)
        images = [img for img in os.listdir(person_dir) if img.endswith(ext)]
        if not images:
            continue

        val_img = np.random.choice(images)
        for img in images:
            src_img = os.path.join(person_dir, img)
            dst_dir = val_path if img == val_img else train_path
            dst_person_dir = os.path.join(dst_dir, person_id)
            os.makedirs(dst_person_dir, exist_ok=True)
            copyfile(src_img, os.path.join(dst_person_dir, img))

def prepare_query_gallery(test_path, save_base, aspect, runs=10, ext='jpg'):
    """Creates query and gallery splits."""
    for run in range(runs):
        query_dir = os.path.join(save_base, f'query{run}')
        gallery_dir = os.path.join(save_base, f'gallery{run}')
        os.makedirs(query_dir, exist_ok=True)
        os.makedirs(gallery_dir, exist_ok=True)

        for person_id in os.listdir(test_path):
            person_dir = os.path.join(test_path, person_id)
            images = [img for img in os.listdir(person_dir) if img.endswith(ext)]
            if not images:
                continue

            gallery_img = np.random.choice(images)
            for img in images:
                src_img = os.path.join(person_dir, img)
                dst_base = gallery_dir if img == gallery_img else query_dir
                dst_person_dir = os.path.join(dst_base, person_id)
                os.makedirs(dst_person_dir, exist_ok=True)
                copyfile(src_img, os.path.join(dst_person_dir, img))

def main(args):
    data_path = args.data_path
    aspects = args.aspects
    train_folder = os.path.join(data_path, 'Hands')
    hand_info_file = os.path.join(data_path, 'HandInfo.csv')

    if not os.path.exists(train_folder) or not os.path.exists(hand_info_file):
        print("Check dataset paths!")
        return

    # Define thresholds for splitting (based on dataset specifics)
    thresholds = {
        'dorsal right': (1050, 1050),
        'dorsal left': (1037, 1037),
        'palmar right': (1051, 1051),
        'palmar left': (1042, 1042)
    }

    # Step 1: Create directory structures
    create_dirs(data_path, aspects, splits=["train_all", "train", "val", "test"])

    # Step 2: Read HandInfo CSV
    hand_data = read_hand_info(hand_info_file)

    # Step 3: Process each aspect
    for aspect in aspects:
        aspect_dir = f"train_val_test_split_{aspect}"
        save_base = os.path.join(data_path, aspect_dir)

        # Thresholds for train/test split (customizable)
        train_thresh, test_thresh = thresholds.get(aspect, (1000, 1000))

        # Split train_all and test
        split_data_by_aspect(hand_data, train_folder, save_base, aspect, train_thresh, test_thresh)

        # Train/Val split
        train_all_path = os.path.join(save_base, "train_all")
        train_path = os.path.join(save_base, "train")
        val_path = os.path.join(save_base, "val")
        split_train_val(train_all_path, train_path, val_path)

        # Query and Gallery splits (default: 10 runs)
        test_path = os.path.join(save_base, "test")
        prepare_query_gallery(test_path, save_base, aspect, runs=args.runs)

    print("\n✅ Data Preparation Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 11k Hand Dataset (Train/Val/Test/Query/Gallery)")
    parser.add_argument('--data_path', type=str, default='./datasets/11khands', help='Path to 11k dataset')
    parser.add_argument('--aspects', nargs='+', default=['dorsal right'], help='Aspects to process (e.g., dorsal right, palmar left)')
    parser.add_argument('--runs', type=int, default=10, help='Number of query/gallery splits (Monte Carlo runs)')

    args = parser.parse_args()
    main(args)
