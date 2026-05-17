"""
Quick YOLOv8 Trainer for PCB Defects
====================================
Simplified training with minimal setup.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    try:
        from ultralytics import YOLO
        import shutil
        
        print("\n" + "="*60)
        print("  TRAINING YOLOv8 FOR PCB DEFECT DETECTION")
        print("="*60 + "\n")
        
        # Check dataset
        dataset_root = Path("./data/pcb_labeled")
        if not (dataset_root / "images").exists():
            print("❌ Dataset not found!")
            return
        
        # Create YOLO dataset structure
        yolo_dataset = Path("./data/yolo_dataset")
        yolo_dataset.mkdir(exist_ok=True)
        (yolo_dataset / "images").mkdir(exist_ok=True)
        (yolo_dataset / "labels").mkdir(exist_ok=True)
        
        for split in ["train", "val", "test"]:
            src_img = dataset_root / "images" / split
            src_lbl = dataset_root / "labels" / split
            dst_img = yolo_dataset / "images" / split
            dst_lbl = yolo_dataset / "labels" / split
            
            dst_img.mkdir(exist_ok=True)
            dst_lbl.mkdir(exist_ok=True)
            
            if src_img.exists():
                for f in src_img.glob("*"):
                    if not (dst_img / f.name).exists():
                        shutil.copy2(f, dst_img / f.name)
            
            if src_lbl.exists():
                for f in src_lbl.glob("*"):
                    if not (dst_lbl / f.name).exists():
                        shutil.copy2(f, dst_lbl / f.name)
        
        print(f"✓ Dataset prepared at {yolo_dataset}")
        
        # Create data.yaml
        yaml_content = f"""path: {str(yolo_dataset.absolute()).replace(chr(92), "/")}
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
        
        yaml_path = yolo_dataset / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"✓ Created {yaml_path}")
        
        # Train model
        print("\n[Training] Loading YOLOv8...")
        model = YOLO("yolov8n.pt")
        
        print("[Training] Starting training...")
        print("  Epochs: 30")
        print("  Batch: 8 (CPU-friendly)")
        print("  Device: CPU")
        
        results = model.train(
            data=str(yaml_path),
            epochs=30,
            imgsz=416,
            batch=8,
            device="cpu",
            patience=5,
            save=True,
            project="./models/phase2",
            name="yolov8_pcb",
            exist_ok=True,
            verbose=False,
            plots=False,
        )
        
        # Save best model
        print("\n[Saving] Copying best model...")
        best_pt = Path("./models/phase2/yolov8_pcb/weights/best.pt")
        if best_pt.exists():
            output_path = Path("./models/phase2/yolov8_pcb_best.pt")
            shutil.copy2(best_pt, output_path)
            print(f"✅ Model saved: {output_path}")
            print(f"   File size: {output_path.stat().st_size / 1e6:.1f} MB")
            
            print("\n" + "="*60)
            print("  ✨ TRAINING COMPLETE!")
            print("="*60)
            print(f"\nYour trained YOLOv8 model is ready!")
            print(f"Location: {output_path}")
            print(f"\nNext steps:")
            print(f"1. The server will use this model automatically")
            print(f"2. Test with: python test_api.py")
            
        else:
            print(f"❌ Best model not found at {best_pt}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
