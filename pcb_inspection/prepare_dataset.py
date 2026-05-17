"""
Dataset Preparation Tool for PCB Inspection
===========================================
Automates the process of adding new datasets to your PCB inspection project.
"""

import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import cv2
from tqdm import tqdm


class DatasetPreparer:
    """Prepares new datasets for PCB inspection training."""

    def __init__(self, project_root="./"):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"

        # Class definitions (same as existing)
        self.classes = {
            0: 'missing_component',
            1: 'solder_bridge',
            2: 'cold_joint',
            3: 'trace_crack',
            4: 'contamination'
        }

    def create_dataset_structure(self, dataset_name):
        """Create the standard dataset directory structure."""
        dataset_path = self.data_dir / dataset_name

        # Create directories
        dirs_to_create = [
            dataset_path / "images" / "train",
            dataset_path / "images" / "val",
            dataset_path / "images" / "test",
            dataset_path / "labels" / "train",
            dataset_path / "labels" / "val",
            dataset_path / "labels" / "test",
            dataset_path / "good"  # For anomaly detection
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ Created dataset structure: {dataset_path}")
        return dataset_path

    def validate_images(self, image_dir):
        """Validate and filter images."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        valid_images = []

        image_path = Path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        for img_file in image_path.glob("*"):
            if img_file.suffix.lower() in valid_extensions:
                # Try to read image to validate
                try:
                    img = cv2.imread(str(img_file))
                    if img is not None and img.size > 0:
                        valid_images.append(img_file)
                    else:
                        print(f"⚠️  Skipping invalid image: {img_file.name}")
                except Exception as e:
                    print(f"⚠️  Error reading {img_file.name}: {e}")

        print(f"✅ Found {len(valid_images)} valid images")
        return valid_images

    def split_dataset(self, image_files, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test sets."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Train/val/test ratios must sum to 1.0")

        # First split: train+val vs test
        train_val_files, test_files = train_test_split(
            image_files,
            test_size=test_ratio,
            random_state=42
        )

        # Second split: train vs val
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=42
        )

        print(f"📊 Split complete:")
        print(f"   Train: {len(train_files)} images")
        print(f"   Val: {len(val_files)} images")
        print(f"   Test: {len(test_files)} images")

        return train_files, val_files, test_files

    def copy_files(self, file_list, src_dir, dst_dir, file_type="images"):
        """Copy files to destination directory."""
        for src_file in tqdm(file_list, desc=f"Copying {file_type}"):
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)

            # Also copy corresponding label file if it exists
            if "images" in file_type:
                label_file = src_file.with_suffix('.txt')
                if label_file.exists():
                    label_dst = dst_dir.parent.parent / "labels" / dst_dir.name / label_file.name
                    shutil.copy2(label_file, label_dst)

    def create_data_yaml(self, dataset_path, dataset_name):
        """Create data.yaml file for YOLO training."""
        data_yaml = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': self.classes
        }

        yaml_path = dataset_path / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"✅ Created data.yaml: {yaml_path}")

    def prepare_anomaly_dataset(self, good_images_dir, dataset_path):
        """Prepare good images for anomaly detection."""
        if not Path(good_images_dir).exists():
            print("⚠️  No good images directory found - skipping anomaly dataset")
            return

        good_images = self.validate_images(good_images_dir)
        if not good_images:
            print("⚠️  No valid good images found")
            return

        # Copy good images
        good_dst = dataset_path / "good"
        self.copy_files(good_images, Path(good_images_dir), good_dst, "good images")

        print(f"✅ Prepared {len(good_images)} good images for anomaly detection")

    def add_to_existing_dataset(self, new_images_dir, new_labels_dir=None):
        """Add images to existing dataset."""
        existing_dataset = self.data_dir / "pcb_labeled"

        if not existing_dataset.exists():
            raise FileNotFoundError("Existing dataset not found. Run prepare_new_dataset first.")

        # Validate new images
        new_images = self.validate_images(new_images_dir)

        # Split new images
        train_files, val_files, test_files = self.split_dataset(new_images)

        # Copy to existing dataset
        base_path = Path(new_images_dir)

        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if files:
                img_dst = existing_dataset / "images" / split_name
                self.copy_files(files, base_path, img_dst, f"{split_name} images")

        print("✅ Added new images to existing dataset")

    def prepare_new_dataset(self, name, images_dir, good_images_dir=None, train_ratio=0.7,
                          val_ratio=0.2, test_ratio=0.1):
        """Prepare a complete new dataset."""
        print(f"\n🔧 Preparing new dataset: {name}")
        print("=" * 50)

        # Create structure
        dataset_path = self.create_dataset_structure(name)

        # Validate and split images
        image_files = self.validate_images(images_dir)
        if not image_files:
            raise ValueError("No valid images found")

        train_files, val_files, test_files = self.split_dataset(
            image_files, train_ratio, val_ratio, test_ratio
        )

        # Copy files
        base_path = Path(images_dir)

        splits = [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]

        for split_name, files in splits:
            if files:
                img_dst = dataset_path / "images" / split_name
                self.copy_files(files, base_path, img_dst, f"{split_name} images")

        # Prepare anomaly detection dataset
        if good_images_dir:
            self.prepare_anomaly_dataset(good_images_dir, dataset_path)

        # Create data.yaml
        self.create_data_yaml(dataset_path, name)

        print(f"\n🎉 Dataset '{name}' prepared successfully!")
        print(f"   Location: {dataset_path}")
        print(f"   Ready for training with: python train_yolo_simple.py --data {dataset_path}/data.yaml")

        return dataset_path


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Prepare PCB datasets for inspection training")
    parser.add_argument("--action", choices=["new", "add"], required=True,
                       help="Create new dataset or add to existing")
    parser.add_argument("--name", help="Dataset name (for new datasets)")
    parser.add_argument("--images", required=True, help="Directory with PCB images")
    parser.add_argument("--good-images", help="Directory with good PCB images (for anomaly detection)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")

    args = parser.parse_args()

    preparer = DatasetPreparer()

    try:
        if args.action == "new":
            if not args.name:
                parser.error("--name is required for new datasets")

            preparer.prepare_new_dataset(
                name=args.name,
                images_dir=args.images,
                good_images_dir=args.good_images,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio
            )

        elif args.action == "add":
            preparer.add_to_existing_dataset(
                new_images_dir=args.images
            )

        print("\n✅ Dataset preparation complete!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())