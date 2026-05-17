"""
Quick Test Script — Verify Local Setup
======================================
Tests the PCB inspection pipeline locally.

Usage:
    python test_local.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    imports = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("anomalib", "Anomalib"),
        ("ultralytics", "Ultralytics"),
        ("onnxruntime", "ONNX Runtime"),
        ("cv2", "OpenCV"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
    ]

    success_count = 0
    for module, name in imports:
        try:
            if module == "cv2":
                import cv2 as cv2_module
                version = cv2_module.__version__
            else:
                module_obj = __import__(module)
                version = getattr(module_obj, "__version__", "unknown")

            print(f"✅ {name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: Import failed - {e}")
        except Exception as e:
            print(f"⚠️ {name}: Loaded but issue - {e}")
            success_count += 1

    print(f"\nImport test: {success_count}/{len(imports)} packages loaded")
    return success_count >= 6  # Require at least basic packages

def test_models():
    """Test loading trained models if they exist."""
    print("\nTesting model loading...")

    # Test Phase 1
    pc_path = Path("./models/phase1/patchcore.onnx")
    if pc_path.exists():
        try:
            from src.inspector import PCBInspector
            inspector = PCBInspector(str(pc_path))
            print("✅ Phase 1 (PatchCore) model loaded successfully")
        except Exception as e:
            print(f"❌ Phase 1 model loading failed: {e}")
    else:
        print("⚠️ Phase 1 model not found (train with: python src/train_phase1.py)")

    # Test Phase 2
    yolo_path = Path("./models/phase2/yolov8m.onnx")
    if yolo_path.exists():
        try:
            from src.inspector import YOLOInspector
            inspector = YOLOInspector(str(yolo_path))
            print("✅ Phase 2 (YOLOv8) model loaded successfully")
        except Exception as e:
            print(f"❌ Phase 2 model loading failed: {e}")
    else:
        print("⚠️ Phase 2 model not found (train with: python src/train_phase2.py)")

def test_data():
    """Check if data directories exist."""
    print("\nChecking data directories...")

    data_checks = [
        ("./data/pcb/good", "Good PCB images"),
        ("./data/pcb_labeled/images/train", "Labeled training images"),
        ("./data/pcb_labeled/labels/train", "Training labels"),
    ]

    for path, description in data_checks:
        if Path(path).exists():
            files = list(Path(path).glob("*"))
            print(f"✅ {description}: {len(files)} files found")
        else:
            print(f"⚠️ {description}: directory not found")

def test_inference():
    """Test a quick inference if models are available."""
    print("\nTesting inference...")

    # Create a dummy image for testing
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    pc_path = Path("./models/phase1/patchcore.onnx")
    yolo_path = Path("./models/phase2/yolov8m.onnx")

    if pc_path.exists():
        try:
            from src.inspector import PCBInspector
            inspector = PCBInspector(str(pc_path))
            result = inspector.inspect(dummy_img)
            print(f"✅ Phase 1 inference: score={result['score']:.3f}, defective={result['defective']}")
        except Exception as e:
            print(f"❌ Phase 1 inference failed: {e}")

    if yolo_path.exists():
        try:
            from src.inspector import YOLOInspector
            inspector = YOLOInspector(str(yolo_path))
            result = inspector.inspect(dummy_img)
            print(f"✅ Phase 2 inference: {len(result)} detections")
        except Exception as e:
            print(f"❌ Phase 2 inference failed: {e}")

def main():
    print("🔍 PCB Inspection Local Setup Test")
    print("=" * 50)

    success = test_imports()
    if not success:
        print("\n❌ Basic imports failed. Run: pip install -r requirements.txt")
        sys.exit(1)

    test_models()
    test_data()
    test_inference()

    print("\n" + "=" * 50)
    print("🎉 Local setup test complete!")
    print("\nNext steps:")
    print("1. Add good PCB images to data/pcb/good/")
    print("2. Train Phase 1: python src/train_phase1.py")
    print("3. Run demo: streamlit run demo.py")
    print("4. Or run API: python src/local_server.py")

if __name__ == "__main__":
    main()