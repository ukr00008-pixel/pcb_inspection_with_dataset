"""
Simplified Phase 1 Training: Basic Anomaly Detection (No Anomalib)
===================================================================
Trains a basic statistical anomaly detector using good PCB images.
No complex ML libraries required - works with any Python version.
"""

import argparse
from pathlib import Path
import pickle
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train basic anomaly detection model")
    parser.add_argument("--data", default="./data/pcb",
                        help="Root data dir containing good/ subfolder")
    parser.add_argument("--output", default="./models/phase1",
                        help="Directory to save trained model")
    return parser.parse_args()


def train(args):
    """Train the simplified anomaly detection model."""
    from simplified_inspector import SimplePCBInspector

    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 1] Data root  : {data_root}")
    print(f"[Phase 1] Output dir : {output_dir}")

    # Check if good images directory exists
    good_images_dir = data_root / "good"
    if not good_images_dir.exists():
        print(f"❌ Good images directory not found: {good_images_dir}")
        print("Please create the directory and add good PCB images.")
        return None

    # Count available images
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(good_images_dir.glob(ext)))

    if not image_files:
        print(f"❌ No image files found in {good_images_dir}")
        print("Supported formats: JPG, JPEG, PNG, BMP")
        return None

    print(f"[Phase 1] Found {len(image_files)} good images")

    # Initialize inspector
    inspector = SimplePCBInspector(threshold=0.5)

    # Train on good images
    print("\n[Phase 1] Training anomaly detector...")
    try:
        reference_stats = inspector.train(str(good_images_dir))
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

    # Save trained model
    model_path = output_dir / "simple_anomaly_detector.pkl"
    try:
        with open(model_path, 'wb') as f:
            pickle.dump({
                'reference_stats': reference_stats,
                'threshold': inspector.threshold,
                'training_images': len(image_files)
            }, f)
        print(f"✅ Model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return None

    # Print summary
    print(f"\n{'='*50}")
    print("  TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"  Training images: {len(image_files)}")
    print(f"  Features learned: {len(reference_stats)}")
    print(f"  Threshold: {inspector.threshold}")
    print(f"  Model saved: {model_path}")
    print(f"{'='*50}")

    return str(model_path)


def main():
    args = parse_args()
    model_path = train(args)

    if model_path:
        print("\n🎉 Phase 1 training completed!")
        print(f"Model saved at: {model_path}")
        print("\nNext steps:")
        print("1. Train Phase 2: python src/train_phase2.py")
        print("2. Run demo: streamlit run demo.py")
        print("3. Or run API: python src/local_server.py")
    else:
        print("\n❌ Training failed. Please check your data and try again.")


if __name__ == "__main__":
    main()