# 📊 Adding New Datasets to PCB Inspection Project

This guide shows you how to add new PCB datasets to your inspection system.

## 📁 Current Dataset Structure

Your project uses two types of datasets:

### 1. **Anomaly Detection Dataset** (`data/pcb/`)
```
data/pcb/
└── good/           # Images of good PCBs (no defects)
    ├── good_0000.jpg
    ├── good_0001.jpg
    └── ...
```

### 2. **Defect Detection Dataset** (`data/pcb_labeled/`)
```
data/pcb_labeled/
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images
└── labels/
    ├── train/      # YOLO format labels (.txt)
    ├── val/        # YOLO format labels (.txt)
    └── test/       # YOLO format labels (.txt)
```

## 🆕 Adding a New Dataset

### Step 1: Prepare Your Images

1. **Collect PCB Images**: Gather high-quality images of PCBs
2. **Organize by Quality**:
   - **Good PCBs**: No defects, used for anomaly detection training
   - **Defective PCBs**: With specific defects for YOLO training

### Step 2: Choose Dataset Type

#### Option A: Add to Existing Dataset
If you want to **extend** the current dataset:

1. **Add Good Images**:
   ```bash
   # Copy new good PCB images to:
   cp new_good_images/*.jpg data/pcb/good/
   ```

2. **Add Labeled Defect Images**:
   ```bash
   # Copy images to appropriate splits:
   cp new_train_images/*.jpg data/pcb_labeled/images/train/
   cp new_val_images/*.jpg data/pcb_labeled/images/val/
   cp new_test_images/*.jpg data/pcb_labeled/images/test/

   # Copy corresponding labels:
   cp new_train_labels/*.txt data/pcb_labeled/labels/train/
   cp new_val_labels/*.txt data/pcb_labeled/labels/val/
   cp new_test_labels/*.txt data/pcb_labeled/labels/test/
   ```

#### Option B: Create New Dataset Directory
If you want a **separate dataset**:

1. **Create New Dataset Structure**:
   ```bash
   # Create new dataset directory
   mkdir -p data/new_pcb_dataset/{images/{train,val,test},labels/{train,val,test}}

   # Add good images for anomaly detection
   mkdir -p data/new_pcb_dataset/good
   cp new_good_images/*.jpg data/new_pcb_dataset/good/
   ```

### Step 3: Label Your Data (For YOLO Training)

#### YOLO Annotation Format
Each label file corresponds to an image and contains:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example** (`defect_001.txt`):
```
2 0.354687 0.334375 0.043750 0.043750  # cold_joint
2 0.635938 0.173437 0.037500 0.037500  # another cold_joint
```

**Class IDs** (from `data.yaml`):
- `0`: missing_component
- `1`: solder_bridge
- `2`: cold_joint
- `3`: trace_crack
- `4`: contamination

#### Labeling Tools:
1. **LabelImg** (Free, open-source)
2. **Labelbox** (Web-based)
3. **CVAT** (Web-based, free)
4. **Roboflow** (Web-based with auto-labeling)

### Step 4: Create/Update Dataset Configuration

#### For New Dataset:
Create `data/new_pcb_dataset/data.yaml`:
```yaml
path: D:/pcb_inspection_with_dataset/pcb_inspection/data/new_pcb_dataset
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
```

#### For Extended Dataset:
The existing `data/yolo_dataset/data.yaml` will automatically include new images.

### Step 5: Update Training Scripts

#### Option A: Use Existing Scripts
Your current scripts will automatically detect new images in the directories.

#### Option B: Create Custom Training Script
```python
# new_dataset_trainer.py
from ultralytics import YOLO
import torch

def train_new_dataset():
    # Load model
    model = YOLO('yolov8n.pt')  # or your trained model

    # Train on new dataset
    results = model.train(
        data='data/new_pcb_dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='new_pcb_model'
    )

    return results

if __name__ == "__main__":
    train_new_dataset()
```

### Step 6: Retrain Models

#### Retrain Anomaly Detection:
```bash
# This will include new good images
python -c "
from src.simplified_inspector import SimplePCBInspector
import pickle

inspector = SimplePCBInspector(threshold=0.1)
inspector.train('data/pcb/good')  # Include your new good images

# Save updated model
with open('models/phase1/simple_anomaly_detector.pkl', 'wb') as f:
    pickle.dump({
        'reference_stats': inspector.reference_stats,
        'threshold': inspector.threshold
    }, f)
"
```

#### Retrain YOLO Model:
```bash
python train_yolo_simple.py
```

### Step 7: Test New Dataset

#### Test with Webcam Inspector:
```bash
python simple_webcam_server.py
# Open http://localhost:8081/webcam
```

#### Test with API:
```python
from src.simplified_inspector import EnsembleInspector

# Load updated models
inspector = EnsembleInspector(
    patchcore_path='models/phase1/simple_anomaly_detector.pkl',
    yolo_path='models/phase2/yolov8_pcb_best.pt'
)

# Test on new image
import cv2
img = cv2.imread('new_test_image.jpg')
results = inspector.inspect(img)
print(f"Pass: {results['pass']}")
print(f"Defects: {len(results['defects'])}")
```

## 📊 Dataset Quality Guidelines

### Image Requirements:
- **Resolution**: 640x640 minimum, 1280x720 recommended
- **Format**: JPEG or PNG
- **Lighting**: Consistent, good illumination
- **Angle**: Top-down view preferred
- **Focus**: Sharp, in-focus images

### Labeling Best Practices:
- **Bounding Boxes**: Tight fit around defects
- **Multiple Defects**: Label all visible defects in one image
- **Class Accuracy**: Double-check class assignments
- **Consistency**: Same labeling standards across all images

### Dataset Size Recommendations:
- **Minimum**: 100 images per class for basic training
- **Good**: 500+ images per class for robust models
- **Excellent**: 1000+ images per class for production use

## 🔧 Automation Scripts

### Dataset Preparation Script:
```python
# prepare_dataset.py
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_dataset(source_dir, target_dir, test_size=0.2, val_size=0.2):
    """Prepare dataset with train/val/test splits."""

    # Create directories
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{target_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{target_dir}/labels/{split}", exist_ok=True)

    # Get all image files
    image_files = list(Path(source_dir).glob("*.jpg")) + \
                  list(Path(source_dir).glob("*.png"))

    # Split dataset
    train_val, test = train_test_split(
        image_files, test_size=test_size, random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=42
    )

    # Copy files
    for split, files in [('train', train), ('val', val), ('test', test)]:
        for img_path in files:
            # Copy image
            shutil.copy2(img_path, f"{target_dir}/images/{split}/{img_path.name}")

            # Copy corresponding label (if exists)
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy2(label_path, f"{target_dir}/labels/{split}/{label_path.name}")

    print(f"Dataset prepared: {len(train)} train, {len(val)} val, {len(test)} test")

# Usage
prepare_dataset("new_pcb_images/", "data/new_dataset/")
```

## 🚀 Quick Start for New Dataset

1. **Collect Images**:
   ```bash
   mkdir new_pcb_data
   # Add your PCB images here
   ```

2. **Label Images**:
   ```bash
   # Use your preferred labeling tool
   # Save labels in YOLO format
   ```

3. **Prepare Dataset**:
   ```bash
   python prepare_dataset.py
   ```

4. **Train Models**:
   ```bash
   python train_yolo_simple.py
   ```

5. **Test**:
   ```bash
   python simple_webcam_server.py
   ```

## ❓ Troubleshooting

### Common Issues:

**"No images found"**
- Check file paths and extensions
- Ensure images are in correct directories

**"Label file missing"**
- Every image needs a corresponding .txt label file
- Even if no defects, create empty .txt file

**"Class ID out of range"**
- Check your data.yaml class definitions
- Ensure label files use correct class IDs (0-4)

**"Training fails"**
- Check dataset YAML paths
- Ensure all required directories exist
- Verify image/label file correspondence

### Getting Help:
- Check existing `train_yolo_simple.py` for reference
- Review `data/yolo_dataset/data.yaml` for format
- Test with small dataset first (10-20 images)

---

**Happy Dataset Building! 📊🔧**