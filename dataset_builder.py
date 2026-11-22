# ========================================================================================
# IndicDLP: Dataset Builder & Stratified Partition Engine (70% Train / 15% Val / 15% Test)
# Target: 10 Batches (Batch_01 to Batch_10) -> Ad_Dataset (next to this script)
# ========================================================================================

import os
import shutil
import random
import json
import sys
from pathlib import Path

# Paths
# Usage: python dataset_builder.py [source_root] [target_root]
# source_root holds Batch_01..Batch_10, each with ad_crops/ and non_ad_crops/.
HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "Detected_Advertisements"
TARGET_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "Ad_Dataset"
MANIFEST_PATH = TARGET_ROOT / "dataset_manifest.json"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def collect_crops():
    ad_files = []
    non_ad_files = []

    for b in range(1, 11):
        batch_folder = SOURCE_ROOT / f"Batch_{b:02d}"
        ad_dir = batch_folder / "ad_crops"
        non_ad_dir = batch_folder / "non_ad_crops"

        if ad_dir.exists():
            for f in ad_dir.glob("*.*"):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    ad_files.append(f)

        if non_ad_dir.exists():
            for f in non_ad_dir.glob("*.*"):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    non_ad_files.append(f)

    return ad_files, non_ad_files

def split_data(files, train_ratio=0.70, val_ratio=0.15):
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    return train_files, val_files, test_files

def main():
    print("=" * 65)
    print("🚀 Initializing Ad Dataset Builder & Stratified Splitter")
    print("=" * 65)

    ad_files, non_ad_files = collect_crops()
    print(f"📊 Collected from Batches 1 to 10:")
    print(f"   • Commercial Ad Crops:     {len(ad_files):,}")
    print(f"   • Editorial Non-Ad Crops:  {len(non_ad_files):,}")
    print(f"   • Total Crops:             {len(ad_files) + len(non_ad_files):,}")

    train_ads, val_ads, test_ads = split_data(ad_files, 0.70, 0.15)
    train_non, val_non, test_non = split_data(non_ad_files, 0.70, 0.15)

    partitions = {
        "train": {"ad": train_ads, "non_ad": train_non},
        "val": {"ad": val_ads, "non_ad": val_non},
        "test": {"ad": test_ads, "non_ad": test_non}
    }

    print("\n📂 Creating Partitioned Dataset Structure on Desktop...")
    manifest = {
        "dataset_name": "IndicDLP Historical Ad vs Non-Ad Benchmark",
        "random_seed": RANDOM_SEED,
        "total_crops": len(ad_files) + len(non_ad_files),
        "classes": ["non_ad", "ad"],
        "class_counts": {
            "ad": len(ad_files),
            "non_ad": len(non_ad_files)
        },
        "partitions": {}
    }

    for split_name, class_dict in partitions.items():
        split_manifest = {}
        for class_name, files in class_dict.items():
            dest_dir = TARGET_ROOT / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            file_list = []
            for src_path in files:
                dest_path = dest_dir / src_path.name
                if not dest_path.exists():
                    shutil.copy2(src_path, dest_path)
                file_list.append(src_path.name)

            split_manifest[class_name] = len(files)
            print(f"   ✓ {split_name.upper():5s} | {class_name:7s}: {len(files):,}")

        manifest["partitions"][split_name] = split_manifest

    # Calculate class weights for training
    train_total = len(train_ads) + len(train_non)
    w_non_ad = round(train_total / (2.0 * len(train_non)), 4)
    w_ad = round(train_total / (2.0 * len(train_ads)), 4)
    manifest["class_weights_train"] = {
        "0_non_ad": w_non_ad,
        "1_ad": w_ad
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print(f"🎉 Dataset Built Successfully at: {TARGET_ROOT}")
    print(f"📄 Manifest Written: {MANIFEST_PATH}")
    print(f"⚖️  Calculated Loss Weights: Non-Ad = {w_non_ad} | Ad = {w_ad}")
    print("=" * 65)

if __name__ == "__main__":
    main()
