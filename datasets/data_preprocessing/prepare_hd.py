"""
prepare_hd.py

Data Preparation Script for HD Hand Dataset.
- Splits HD dataset into train, val, test, query, and gallery folders.
- Adds additional 211 subjects to gallery for testing (as in the original script).
"""

import os
import numpy as np
import argparse
from shutil import copyfile


def create_dirs(save_path, query_gallery_splits=10):
    """Create base directories for train/val/test and query/gallery splits."""
    paths = ['train_all', 'train', 'val', 'test']
    for folder in paths:
        dir_path = os.path.join(save_path, folder)
        os.makedirs(dir_path, exist_ok=True)

    for i in range(query_gallery_splits):
        query_path = os.path.join(save_path, f'query{i}')
        gallery_path = os.path.join(save_path, f'gallery{i}')
        os.makedirs(query_path, exist_ok=True)
        os.makedirs(gallery_path, exist_ok=True)


def split_train_val(train_path, train_all_save_path, train_save_path, val_save_path, ext):
    """Splits train_all into train and val sets."""
    for dir_name in os.listdir(train_all_save_path):
        dir_full_path = os.path.join(train_all_save_path, dir_name)
        files = [f for f in os.listdir(dir_full_path) if f.endswith(ext)]
        if not files:
            continue

        val_ind = np.random.randint(0, len(files))
        val_file = files[val_ind]

        for file in files:
            src_file = os.path.join(dir_full_path, file)
            if file == val_file:
                dst_dir = os.path.join(val_save_path, dir_name)
            else:
                dst_dir = os.path.join(train_save_path, dir_name)
            os.makedirs(dst_dir, exist_ok=True)
            copyfile(src_file, os.path.join(dst_dir, file))


def prepare_query_gallery(test_save_path, add2gallery_path, save_path, ext, runs=10, random_gallery=True):
    """Prepares query and gallery sets with optional extra gallery subjects."""
    for i in range(runs):
        query_save_path = os.path.join(save_path, f'query{i}')
        gallery_save_path = os.path.join(save_path, f'gallery{i}')

        # Generate query and gallery splits
        for dir_name in os.listdir(test_save_path):
            dir_full_path = os.path.join(test_save_path, dir_name)
            files = [f for f in os.listdir(dir_full_path) if f.endswith(ext)]
            if not files:
                continue

            val_ind = np.random.randint(0, len(files))
            val_file = files[val_ind]

            for file in files:
                src_file = os.path.join(dir_full_path, file)
                if random_gallery and file == val_file:
                    dst_dir = os.path.join(gallery_save_path, dir_name)
                else:
                    dst_dir = os.path.join(query_save_path, dir_name)
                os.makedirs(dst_dir, exist_ok=True)
                copyfile(src_file, os.path.join(dst_dir, file))

        # Add 211 (actually 213) additional subjects to gallery
        for file in os.listdir(add2gallery_path):
            if not file.endswith(ext):
                continue
            ID = file.split('_')[0]
            ID_new = str(int(ID) + 1000)  # Offset for new subjects
            src_file = os.path.join(add2gallery_path, file)
            dst_dir = os.path.join(gallery_save_path, ID_new)
            os.makedirs(dst_dir, exist_ok=True)
            copyfile(src_file, os.path.join(dst_dir, file))


def split_train_all_and_test(train_path, train_all_save_path, test_save_path, ext):
    """Splits train_path into train_all and test sets."""
    for file in os.listdir(train_path):
        if not file.endswith(ext):
            continue
        ID = file.split('_')[0]
        src_file = os.path.join(train_path, file)

        if int(ID) <= 447:
            dst_dir = os.path.join(train_all_save_path, ID)
        else:
            dst_dir = os.path.join(test_save_path, ID)
        os.makedirs(dst_dir, exist_ok=True)
        copyfile(src_file, os.path.join(dst_dir, file))


def main(args):
    data_path = args.data_path
    ext = args.ext
    runs = args.runs
    random_gallery = args.random_gallery

    # HD Dataset structure
    train_path = os.path.join(data_path, '1-501')  # 501 identities
    add2gallery_path = os.path.join(data_path, '502-712')  # Additional 211 identities
    save_path = os.path.join(data_path, 'train_val_test_split')

    # Validate paths
    assert os.path.isdir(train_path), f"Train path not found: {train_path}"
    assert os.path.isdir(add2gallery_path), f"Additional gallery path not found: {add2gallery_path}"

    # Step 1: Create base folders
    create_dirs(save_path, query_gallery_splits=runs)

    print("========== Starting data preparation ==========")

    # Step 2: Split train_path into train_all and test sets
    split_train_all_and_test(train_path,
                             os.path.join(save_path, 'train_all'),
                             os.path.join(save_path, 'test'),
                             ext)

    # Step 3: Split train_all into train and val
    split_train_val(train_path,
                    os.path.join(save_path, 'train_all'),
                    os.path.join(save_path, 'train'),
                    os.path.join(save_path, 'val'),
                    ext)

    # Step 4: Prepare query/gallery splits + add additional gallery identities
    prepare_query_gallery(os.path.join(save_path, 'test'),
                          add2gallery_path,
                          save_path,
                          ext,
                          runs=runs,
                          random_gallery=random_gallery)

    print("✅ Data preparation for HD dataset completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HD Hand Dataset (Train/Val/Test/Query/Gallery Splits)")
    parser.add_argument('--data_path', type=str, default='./datasets/HD/Original Images',
                        help='Path to HD dataset folder')
    parser.add_argument('--ext', type=str, default='jpg', help='Image extension')
    parser.add_argument('--runs', type=int, default=10, help='Number of Monte Carlo query/gallery runs')
    parser.add_argument('--random_gallery', action='store_true', help='Randomly select gallery images')

    args = parser.parse_args()
    main(args)
