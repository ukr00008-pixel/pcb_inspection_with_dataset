"""
Train YOLOv8 Model for PCB Defect Detection (Phase 2)
======================================================
Trains YOLOv8 nano model on labeled PCB defects.
Uses CPU for training - should complete in 10-20 minutes depending on system.
"""

import argparse
from pathlib import Path
import os
from datetime import datetime


def setup_dataset():
    """
    Prepare the dataset structure for YOLOv8 training.
    YOLOv8 expects:
    dataset/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
    """
    print("[Phase 2] Setting up YOLO dataset structure...")
    
    dataset_root = Path("./data/yolo_dataset")
    dataset_root.mkdir(exist_ok=True)
    
    # Create directory structure
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # Copy files from existing structure
    import shutil
    
    source_images = Path("./data/pcb_labeled/images")
    source_labels = Path("./data/pcb_labeled/labels")
    
    # Copy train split
    for split in ["train", "val", "test"]:
        src_img = source_images / split
        dst_img = dataset_root / "images" / split
        src_lbl = source_labels / split
        dst_lbl = dataset_root / "labels" / split
        
        if src_img.exists():
            for file in src_img.glob("*"):
                shutil.copy2(file, dst_img / file.name)
                print(f"  Copied {file.name} to {split}")
        
        if src_lbl.exists():
            for file in src_lbl.glob("*"):
                shutil.copy2(file, dst_lbl / file.name)
    
    return dataset_root


def create_dataset_yaml(dataset_root):
    """Create the YAML file for YOLO dataset configuration."""
    yaml_content = """path: """ + str(dataset_root.absolute()).replace("\\", "/") + """
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
    
    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"[Phase 2] Created dataset YAML: {yaml_path}")
    return str(yaml_path)


def train_yolov8(yaml_path, output_dir):
    """
    Train YOLOv8 nano model on PCB defect dataset.
    Uses CPU - will take longer but no GPU needed.
    """
    from ultralytics import YOLO
    
    print("[Phase 2] Starting YOLOv8 Training...")
    print("  Model: YOLOv8-nano (fast, lightweight)")
    print("  Device: CPU (adjust epochs if too slow)")
    print("  Epochs: 50 (for demo; increase to 100+ for production)")
    
    # Load the pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=50,                    # Use fewer epochs for demo (25-50)
        imgsz=416,                     # Image size
        batch=8,                       # Small batch for CPU
        device=0 if cuda_available() else "cpu",  # Use CPU if no CUDA
        patience=10,                   # Early stopping patience
        save=True,
        project="./models/phase2",
        name="pcb_yolov8",
        exist_ok=True,
        verbose=True,
        plots=False,                   # Skip plotting for headless environment
        half=False,                    # No half precision on CPU
    )
    
    return results


def cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def save_best_model(phase2_dir, output_path):
    """Copy the best trained model to the standard location."""
    import shutil
    
    # Find the best model
    best_model = phase2_dir / "pcb_yolov8" / "weights" / "best.pt"
    
    if best_model.exists():
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, output_path)
        print(f"✅ Best model saved to: {output_path}")
        return True
    else:
        print(f"❌ Best model not found at: {best_model}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for PCB defect detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce for low RAM)")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  PCB DEFECT DETECTION - PHASE 2: YOLOv8 TRAINING")
    print("="*60)
    
    try:
        # Setup dataset
        dataset_root = setup_dataset()
        yaml_path = create_dataset_yaml(dataset_root)
        
        # Check if PyTorch/YOLOv8 is available
        try:
            from ultralytics import YOLO
        except ImportError:
            print("\n⚠️  YOLOv8 (ultralytics) not installed!")
            print("Install with: pip install ultralytics")
            return
        
        # Train model
        results = train_yolov8(yaml_path, "./models/phase2")
        
        # Save model
        phase2_dir = Path("./models/phase2")
        output_model = "./models/phase2/yolov8_pcb_best.pt"
        save_best_model(phase2_dir, output_model)
        
        # Print summary
        print(f"\n{'='*60}")
        print("  TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"  Model: YOLOv8-nano")
        print(f"  Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Dataset: {dataset_root.absolute()}")
        print(f"  Output: {output_model}")
        print(f"{'='*60}\n")
        
        print("🎉 Phase 2 training completed!")
        print("Next steps:")
        print("1. Update local_server.py to use the new YOLO model")
        print("2. Restart the server: python src/local_server.py")
        print("3. Test with: python test_api.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
