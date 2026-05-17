"""
Example: Adding New Dataset to PCB Inspection Project
=====================================================
This script demonstrates how to add a new dataset to your PCB inspection system.
"""

import os
from pathlib import Path
from prepare_dataset import DatasetPreparer

def example_add_new_dataset():
    """Example of adding a completely new dataset."""

    print("🆕 Example: Adding New Dataset")
    print("=" * 40)

    # Initialize preparer
    preparer = DatasetPreparer()

    # Example: Prepare a new dataset
    # Assume you have images in "my_new_pcb_images/" directory
    # and good images in "my_good_pcbs/" directory

    try:
        dataset_path = preparer.prepare_new_dataset(
            name="my_custom_pcb_dataset",
            images_dir="my_new_pcb_images",  # Your new PCB images with defects
            good_images_dir="my_good_pcbs",  # Good PCB images for anomaly detection
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )

        print(f"\n✅ New dataset created at: {dataset_path}")
        print("\nNext steps:")
        print("1. Review the dataset structure")
        print("2. Train YOLO model: python train_yolo_simple.py")
        print("3. Test with: python simple_webcam_server.py")

    except Exception as e:
        print(f"❌ Error: {e}")

def example_extend_existing_dataset():
    """Example of adding images to existing dataset."""

    print("\n➕ Example: Extending Existing Dataset")
    print("=" * 40)

    preparer = DatasetPreparer()

    try:
        # Add new images to existing dataset
        preparer.add_to_existing_dataset(
            new_images_dir="additional_pcb_images"
        )

        print("✅ Images added to existing dataset")
        print("Next: Retrain models to include new data")

    except Exception as e:
        print(f"❌ Error: {e}")

def example_manual_setup():
    """Example of manual dataset setup."""

    print("\n🔧 Manual Dataset Setup Example")
    print("=" * 40)

    # Create directory structure manually
    dataset_name = "manual_pcb_dataset"
    base_dir = Path("./data") / dataset_name

    print("Creating directory structure...")

    # Create all required directories
    dirs = [
        base_dir / "images" / "train",
        base_dir / "images" / "val",
        base_dir / "images" / "test",
        base_dir / "labels" / "train",
        base_dir / "labels" / "val",
        base_dir / "labels" / "test",
        base_dir / "good"
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d}")

    # Create data.yaml
    data_yaml = f"""
path: {base_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 5
names:
  0: missing_component
  1: solder_bridge
  2: cold_joint
  3: trace_crack
  4: contamination
"""

    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_yaml.strip())

    print(f"✅ Created data.yaml: {yaml_path}")
    print("Now manually copy your files:")
    print(f"  Images → {base_dir}/images/[train|val|test]/")
    print(f"  Labels → {base_dir}/labels/[train|val|test]/")
    print(f"  Good images → {base_dir}/good/")

def show_directory_structure():
    """Show the expected directory structure."""

    print("\n📁 Expected Dataset Structure")
    print("=" * 40)

    structure = """
data/
├── your_dataset_name/
│   ├── data.yaml                    # Dataset configuration
│   ├── images/
│   │   ├── train/                   # Training images
│   │   ├── val/                     # Validation images
│   │   └── test/                    # Test images
│   ├── labels/
│   │   ├── train/                   # YOLO labels (.txt)
│   │   ├── val/                     # YOLO labels (.txt)
│   │   └── test/                    # YOLO labels (.txt)
│   └── good/                        # Good images for anomaly detection
│
└── pcb_labeled/                     # Existing dataset
    └── ... (same structure)
"""

    print(structure)

def main():
    """Run examples."""
    print("🚀 PCB Dataset Addition Examples")
    print("=" * 50)

    show_directory_structure()

    # Uncomment the examples you want to run:

    # Example 1: Add completely new dataset
    # example_add_new_dataset()

    # Example 2: Extend existing dataset
    # example_extend_existing_dataset()

    # Example 3: Manual setup
    # example_manual_setup()

    print("\n💡 To use these examples:")
    print("1. Create directories with your PCB images")
    print("2. Uncomment the desired example function")
    print("3. Run: python add_dataset_example.py")
    print("\n📖 For detailed instructions, see: ADD_NEW_DATASET.md")

if __name__ == "__main__":
    main()