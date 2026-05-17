"""
🎯 QUICK START: Adding New Dataset to PCB Inspection
====================================================

This script shows you exactly how to add a new dataset in 3 simple steps.
"""

import os
from pathlib import Path

def show_current_structure():
    """Show current dataset structure."""
    print("📁 Current Dataset Structure:")
    print("data/")
    print("├── pcb/                    # Anomaly detection")
    print("│   └── good/              # Good PCB images")
    print("├── pcb_labeled/           # YOLO defect detection")
    print("│   ├── images/{train,val,test}/")
    print("│   └── labels/{train,val,test}/")
    print("└── yolo_dataset/          # Processed dataset")
    print()

def step_by_step_guide():
    """Show step-by-step guide."""
    print("🚀 3-Step Guide to Add New Dataset:")
    print("=" * 50)
    print()

    print("STEP 1: Prepare Your Images")
    print("-" * 30)
    print("📸 Collect PCB images (JPEG/PNG)")
    print("📁 Organize into directories:")
    print("   • defective_images/     # PCBs with defects")
    print("   • good_images/          # Perfect PCBs (optional)")
    print("   • labels/              # YOLO format labels (.txt)")
    print()

    print("STEP 2: Run Dataset Preparation")
    print("-" * 30)
    print("🔧 Automated method:")
    print("   python prepare_dataset.py --action new --name my_dataset --images defective_images/")
    print()
    print("🔧 Manual method:")
    print("   mkdir -p data/my_dataset/{images/{train,val,test},labels/{train,val,test}}")
    print("   # Copy your files to appropriate directories")
    print()

    print("STEP 3: Train and Test")
    print("-" * 30)
    print("🎓 Train models:")
    print("   python train_yolo_simple.py")
    print()
    print("🧪 Test with webcam:")
    print("   python simple_webcam_server.py")
    print("   → Open http://localhost:8081/webcam")
    print()

def example_usage():
    """Show practical example."""
    print("💡 Practical Example:")
    print("=" * 30)
    print()
    print("Assume you have:")
    print("• 500 PCB images with defects in: /home/user/new_pcb_images/")
    print("• 100 good PCB images in: /home/user/good_pcbs/")
    print()

    print("Commands to run:")
    print("1. cd /path/to/pcb_inspection")
    print("2. python prepare_dataset.py \\")
    print("     --action new \\")
    print("     --name custom_pcb_v1 \\")
    print("     --images /home/user/new_pcb_images/ \\")
    print("     --good-images /home/user/good_pcbs/")
    print()
    print("3. python train_yolo_simple.py")
    print("4. python simple_webcam_server.py")
    print()

def troubleshooting():
    """Show common issues and solutions."""
    print("🔧 Troubleshooting:")
    print("=" * 30)
    print()
    print("❌ 'No images found'")
    print("   → Check file paths and extensions (.jpg, .png)")
    print("   → Ensure images are readable by OpenCV")
    print()

    print("❌ 'Label file missing'")
    print("   → Every image needs a .txt file with same name")
    print("   → Format: class_id x_center y_center width height")
    print()

    print("❌ 'Training fails'")
    print("   → Check data.yaml paths are correct")
    print("   → Ensure train/val/test directories exist")
    print()

    print("❌ 'Model not loading'")
    print("   → Check model files exist in models/ directory")
    print("   → Retrain models after adding new data")
    print()

def main():
    """Main guide."""
    print("🎯 How to Add New Dataset to PCB Inspection")
    print("=" * 60)
    print()

    show_current_structure()
    step_by_step_guide()
    example_usage()
    troubleshooting()

    print("📚 Additional Resources:")
    print("• ADD_NEW_DATASET.md      - Complete guide")
    print("• add_dataset_example.py  - Code examples")
    print("• prepare_dataset.py      - Dataset preparation tool")
    print()

    print("🎉 Ready to add your dataset!")
    print("   Start with: python prepare_dataset.py --help")

if __name__ == "__main__":
    main()