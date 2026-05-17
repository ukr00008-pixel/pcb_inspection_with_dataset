#!/usr/bin/env python3
"""
Monitor YOLOv8 training progress.
Run in a separate terminal while training is happening.
"""

import time
import os
from pathlib import Path

def monitor_training():
    models_path = Path("./models/phase2")
    prev_size = 0
    
    print("📊 YOLOv8 Training Monitor")
    print("=" * 50)
    print("Watching for training output...")
    print()
    
    while True:
        if models_path.exists():
            # Look for recent training files
            pt_files = list(models_path.rglob("*.pt"))
            if pt_files:
                latest = max(pt_files, key=lambda p: p.stat().st_mtime)
                size_mb = latest.stat().st_size / 1e6
                print(f"✓ Model found: {latest.name}")
                print(f"  Size: {size_mb:.1f} MB")
                print(f"  Updated: {time.ctime(latest.stat().st_mtime)}")
                break
        
        time.sleep(5)
        print(".", end="", flush=True)
    
    print("\n✅ Training detected!")

if __name__ == "__main__":
    monitor_training()
