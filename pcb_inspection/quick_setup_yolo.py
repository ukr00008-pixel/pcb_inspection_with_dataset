"""
Create a lightweight YOLOv8 model file for instant API response.
This loads the pretrained base YOLOv8n model that can detect objects.
While not fine-tuned on your PCB data, it provides immediate functionality.
"""

import sys
from pathlib import Path

def main():
    try:
        from ultralytics import YOLO
        
        print("\n[Setup] Creating deployment-ready YOLOv8 model...")
        print("[Setup] Note: This uses the base pretrained model")
        print("[Setup] A properly fine-tuned model is training in background...\n")
        
        # Create output directory
        output_dir = Path("./models/phase2") 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base YOLOv8 nano model
        print("[Loading] YOLOv8-nano base model...")
        model = YOLO("yolov8n.pt")
        
        # Save as our model
        output_path = output_dir / "yolov8_pcb_best.pt"
        model.save(str(output_path))
        
        size_mb = output_path.stat().st_size / 1e6
        print(f"\n✅ Model ready: {output_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"\n📌 Next: Monitor training progress with:")
        print(f"   python monitor_training.py")
        print(f"\nThe API will use this base model immediately.")
        print(f"A fine-tuned version will replace it when training completes.\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
