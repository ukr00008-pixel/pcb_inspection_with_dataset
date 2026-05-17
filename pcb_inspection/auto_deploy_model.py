"""
Auto-setup: Move trained YOLOv8 model to deployment location
Run this after training completes (or schedule it)
"""

import time
import shutil
from pathlib import Path

def setup_trained_model():
    """Find and deploy the trained model."""
    
    # Path to find trained model
    runs_dir = Path("./runs/detect/models/phase2")
    expected_best = None
    
    # Check current runs directory
    if runs_dir.exists():
        for pt_file in runs_dir.rglob("best.pt"):
            expected_best = pt_file
            break
    
    # Also check models directory  
    if expected_best is None:
        models_dir = Path("./models/phase2")
        if models_dir.exists():
            for pt_file in models_dir.rglob("best.pt"):
                expected_best = pt_file
                break
    
    if expected_best and expected_best.exists():
        output_path = Path("./models/phase2/yolov8_pcb_best.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Trained model found: {expected_best}")
        print(f"📦 Deploying to: {output_path}")
        
        shutil.copy2(expected_best, output_path)
        
        size_mb = output_path.stat().st_size / 1e6
        print(f"✅ Model ready! ({size_mb:.1f} MB)")
        print(f"\nRestart the server for automatic use:")
        print(f"  python src/local_server.py")
        return True
    else:
        print("⏳ Trained model not found yet. Training may still be running.")
        return False

if __name__ == "__main__":
    # Try immediately
    if not setup_trained_model():
        # If not found, keep checking
        print("\nWaiting and retrying...") 
        for i in range(30):  # Check for up to 30 minutes
            time.sleep(60)
            print(f"[{i+1}/30] Checking for trained model...")
            if setup_trained_model():
                break
