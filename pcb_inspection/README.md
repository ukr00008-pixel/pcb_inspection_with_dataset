# 🔍 PCB Defect Detection System

<div align="center">

**AI-Powered PCB Anomaly Detection using PatchCore + YOLOv8 Ensemble**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-purple.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Real-time PCB defect detection • CPU-only • No Docker required • Production-ready*

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Technologies Used](#️-technologies-used)
- [📋 Prerequisites](#-prerequisites)
- [🚀 Quick Start](#-quick-start)
- [📖 Usage Guide](#-usage-guide)
- [🔧 API Reference](#-api-reference)
- [📁 Project Structure](#-project-structure)
- [🎓 Training Guide](#-training-guide)
- [📊 Evaluation](#-evaluation)
- [🔍 Demo Options](#-demo-options)
- [📁 Adding New Datasets](#-adding-new-datasets)
- [⚙️ Configuration](#️-configuration)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [📞 Contact](#-contact)

---

## 🎯 Overview

This project implements a **two-phase AI-powered PCB defect detection system** that combines unsupervised and supervised machine learning approaches for accurate, real-time inspection of printed circuit boards (PCBs).

### **Phase 1: Statistical Anomaly Detection**
- Uses **traditional computer vision** techniques
- Learns statistical properties from "good" PCB images
- Detects anomalies using feature analysis (edges, colors, textures)
- **No complex ML models required** - works with basic statistics

### **Phase 2: Supervised Defect Classification**
- Employs **YOLOv8** for precise defect localization and classification
- Identifies specific defect types with bounding boxes
- Requires labeled defect dataset for training

### **Ensemble Approach**
- Combines both models for maximum accuracy
- Phase 1 acts as safety net for unknown defects
- Phase 2 provides detailed classification

**Defect Classes Detected:**
- 🔧 `missing_component` - Absent ICs, wrong parts, rotated components
- ⚡ `solder_bridge` - Solder bridging adjacent pads/traces
- ❄️ `cold_joint` - Dull/grainy solder, insufficient wetting
- 🛣️ `trace_crack` - Broken traces, lifted pads, open circuits
- 🧹 `contamination` - Foreign particles, flux residue, moisture

---

## ✨ Features

### 🔬 **AI & Machine Learning**
- ✅ Two-phase anomaly detection (unsupervised + supervised)
- ✅ Real-time inference (< 2 seconds per image)
- ✅ Ensemble decision making
- ✅ Anomaly heatmaps for visualization
- ✅ Confidence scoring for all detections

### 💻 **Development & Deployment**
- ✅ **CPU-only execution** (no GPU required)
- ✅ **No Docker dependency** (runs directly on your system)
- ✅ Multiple deployment options (Web API, Streamlit app, CLI)
- ✅ Production-ready FastAPI server
- ✅ Interactive Streamlit web interface

### 📊 **Data & Training**
- ✅ Automated data preprocessing
- ✅ Data augmentation for defect classes
- ✅ Class balance monitoring
- ✅ Model export to ONNX format
- ✅ Comprehensive evaluation metrics

### 🎨 **User Interface**
- ✅ Drag-and-drop image upload
- ✅ Real-time results display
- ✅ Visual defect annotations
- ✅ Anomaly heatmap overlay
- ✅ RESTful API for integration

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PCB Image     │───▶│   Preprocessing  │───▶│  Phase 1 Model  │
│   (JPEG/PNG)    │    │   (Resize,       │    │  (PatchCore)    │
└─────────────────┘    │    Normalize)    │    │                 │
                       └──────────────────┘    │ • Anomaly Score │
                                               │ • Heatmap       │
                                               └─────────┬───────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   PCB Image     │───▶│   Preprocessing  │───▶│  Phase 2 Model  │
│   (JPEG/PNG)    │    │   (Resize,       │    │   (YOLOv8)      │
└─────────────────┘    │    Normalize)    │    │                 │
                       └──────────────────┘    │ • Defect Boxes  │
                                               │ • Classes       │
                                               │ • Confidence    │
                                               └─────────┬───────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │   Ensemble      │
                                               │   Decision      │
                                               │                 │
                                               │ • PASS/FAIL     │
                                               │ • Defect List   │
                                               │ • Combined Score│
                                               └─────────────────┘
```

---

## 🛠️ Technologies Used

### **Core AI Frameworks**
- **PyTorch** (2.0+) - Deep learning framework
- **TorchVision** - Computer vision utilities
- **Ultralytics YOLOv8** - Object detection framework
- **OpenCV** (4.8+) - Computer vision and image processing
- **NumPy** - Numerical computing for statistical analysis

### **Computer Vision**
- **OpenCV** (4.8+) - Image processing and computer vision
- **Pillow** - Image manipulation
- **Albumentations** - Data augmentation
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing

### **Web & API**
- **FastAPI** - Modern web API framework
- **Uvicorn** - ASGI server for FastAPI
- **Streamlit** - Web app framework for data science

### **Model Optimization**
- **ONNX Runtime** - High-performance inference engine
- **ONNX** - Open Neural Network Exchange format

### **Development Tools**
- **Python** (3.8+) - Core programming language
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning utilities
- **Requests** - HTTP library

---

## 📋 Prerequisites

### **System Requirements**
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python Version**: 3.8 or higher (3.11 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for models and data
- **Processor**: Any modern CPU (Intel i5/i7 or equivalent)

### **Software Dependencies**
- Python 3.8+
- pip package manager
- Git (optional, for cloning)

---

## 🚀 Quick Start

### **1. Clone & Setup**
```bash
# Navigate to project directory
cd pcb_inspection

# Install dependencies
pip install -r requirements.txt
```

### **2. Test Installation**
```bash
# Verify all packages are installed correctly
python test_local.py
```

### **3. Run Basic Demo**
```bash
# See the image processing pipeline in action
python simple_demo.py
```

### **4. Launch Interactive Demo**
```bash
# Open web interface for PCB inspection
streamlit run demo.py
```
*Browser opens at `http://localhost:8501`*

### **5. Start API Server**
```bash
# Launch REST API for programmatic access
python src/local_server.py
```
*API available at `http://localhost:8000`*

---

## 📖 Usage Guide

### **Method 1: Streamlit Web Interface (Recommended)**

```bash
streamlit run demo.py
```

**Features:**
- Drag-and-drop image upload
- Real-time defect detection
- Visual results with heatmaps
- No coding required

### **Method 2: REST API**

```bash
# Start the server
python src/local_server.py

# Test with curl
curl -X POST http://localhost:8000/inspect \
  -F "file=@your_pcb_image.jpg"
```

### **Method 3: Python Script**

```python
from src.inspector import EnsembleInspector

# Load models
inspector = EnsembleInspector(
    patchcore_path="./models/phase1/patchcore.onnx",
    yolo_path="./models/phase2/yolov8m.onnx"
)

# Inspect image
result = inspector.inspect("path/to/pcb_image.jpg")
print(f"PASS: {result['pass']}")
print(f"Defects: {result['defects']}")
```

---

## 🔧 API Reference

### **Base URL**
```
http://localhost:8000
```

### **Endpoints**

#### **GET /health**
Check server status and model loading.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0"
}
```

#### **POST /inspect**
Inspect a PCB image for defects.

**Parameters:**
- `file` (FormData): PCB image file (JPEG/PNG)

**Response:**
```json
{
  "pass": false,
  "defects": [
    {
      "class": "solder_bridge",
      "confidence": 0.87,
      "bbox": [100, 200, 150, 250]
    }
  ],
  "anomaly_score": 0.73,
  "latency_ms": 245.2
}
```

**Example Usage:**
```bash
curl -X POST http://localhost:8000/inspect \
  -F "file=@pcb_board.jpg"
```

---

## 📁 Project Structure

```
pcb_inspection/
├── 📄 README.md                 # This file
├── 📄 requirements.txt          # Python dependencies
├── 📄 demo.py                   # Streamlit web interface
├── 📄 simple_demo.py            # Basic image processing demo
├── 📄 test_local.py             # Setup verification script
│
├── 🔧 configs/
│   └── 📄 pcb.yaml             # YOLOv8 dataset configuration
│
├── 📊 data/
│   ├── 📁 pcb/
│   │   ├── 📁 good/            # Normal PCB images (200+ recommended)
│   │   └── 📁 defects/         # Defect images (optional)
│   └── 📁 pcb_labeled/
│       ├── 📁 images/
│       │   ├── 📁 train/       # Training images with defects
│       │   ├── 📁 val/         # Validation images
│       │   └── 📁 test/        # Test images
│       └── 📁 labels/          # YOLO format annotations
│
├── 🤖 models/                   # Trained model storage
│   ├── 📁 phase1/              # PatchCore models
│   └── 📁 phase2/              # YOLOv8 models
│
├── 📈 results/                  # Evaluation outputs
│   └── 📁 Patchcore/
│       └── 📁 pcb_inspection/
│
└── 🐍 src/                      # Source code
    ├── 📄 __init__.py
    ├── 📄 inspector.py          # Core AI inspection logic
    ├── 📄 local_server.py      # FastAPI server
    ├── 📄 train_phase1.py       # PatchCore training script
    ├── 📄 train_phase2.py       # YOLOv8 training script
    ├── 📄 evaluate.py           # Model evaluation
    ├── 📄 monitor.py            # Dataset analysis
    ├── 📄 preprocessing.py      # Image preprocessing
    ├── 📄 augment_defects.py    # Data augmentation
    └── 📄 server.py             # Original server (with Redis)
```

---

## 🎓 Training Guide

### **Phase 1: Unsupervised Training (PatchCore)**

**Requirements:**
- 200-500 images of good PCBs
- No defect labels required

**Steps:**
```bash
# 1. Place good PCB images in data/pcb/good/
# 2. Train the model
python src/train_phase1_simple.py

# Optional: Customize training
python src/train_phase1_simple.py \
  --data ./data/pcb \
  --output ./models/phase1
```

**Output:** `models/phase1/simple_anomaly_detector.pkl`

### **Phase 2: Supervised Training (YOLOv8)**

**Requirements:**
- 200+ labeled defect images per class
- YOLO format annotations

**Steps:**
```bash
# 1. Prepare labeled dataset in data/pcb_labeled/
# 2. Check class balance
python src/monitor.py --check-balance ./data/pcb_labeled/labels/train

# 3. Optionally augment data
python src/augment_defects.py \
  --images data/pcb_labeled/images/train \
  --labels data/pcb_labeled/labels/train \
  --output data/pcb_labeled_aug \
  --multiplier 5

# 4. Train the model
python src/train_phase2.py

# Optional: Customize training
python src/train_phase2.py \
  --epochs 50 \
  --batch 4 \
  --imgsz 416
```

**Output:** `models/phase2/yolov8m.onnx`

---

## 📊 Evaluation

### **Evaluate Ensemble Performance**
```bash
python src/evaluate.py \
  --phase1 models/phase1/patchcore.onnx \
  --phase2 models/phase2/yolov8m.onnx \
  --test-dir data/pcb_labeled/images/test
```

### **Performance Metrics**
- **Image AUROC**: Area Under Receiver Operating Characteristic
- **Pixel AUROC**: Pixel-level anomaly detection accuracy
- **mAP**: Mean Average Precision for defect detection
- **Inference Latency**: Processing time per image

---

## � Adding New Datasets

Your PCB inspection system supports easy addition of new datasets for improved accuracy and domain adaptation.

### Quick Add (Automated)

#### Option 1: Command Line
```bash
# Linux/Mac
./add_dataset.sh my_dataset /path/to/pcb/images /path/to/good/images

# Windows
add_dataset.bat my_dataset C:\path\to\pcb\images C:\path\to\good\images
```

#### Option 2: Python Script
```bash
python prepare_dataset.py --action new --name my_dataset --images /path/to/images
```

### Manual Addition

1. **Create Dataset Structure**:
   ```bash
   mkdir -p data/my_dataset/{images/{train,val,test},labels/{train,val,test},good}
   ```

2. **Add Your Images**:
   - `images/train/` - Training PCB images
   - `images/val/` - Validation images  
   - `images/test/` - Test images
   - `good/` - Good PCB images (for anomaly detection)

3. **Add Labels** (YOLO format):
   - Label files must match image names: `image_001.jpg` → `image_001.txt`
   - Format: `<class_id> <x_center> <y_center> <width> <height>`
   - Classes: 0=missing_component, 1=solder_bridge, 2=cold_joint, 3=trace_crack, 4=contamination

4. **Create data.yaml**:
   ```yaml
   path: ./data/my_dataset
   train: images/train
   val: images/val
   test: images/test
   nc: 5
   names: ['missing_component', 'solder_bridge', 'cold_joint', 'trace_crack', 'contamination']
   ```

### Extend Existing Dataset

Add images to your current dataset:
```bash
python prepare_dataset.py --action add --images /path/to/new/images
```

### Training with New Data

```bash
# Retrain YOLO model
python train_yolo_simple.py

# Retrain anomaly detector
python -c "
from src.simplified_inspector import SimplePCBInspector
import pickle
inspector = SimplePCBInspector(0.1)
inspector.train('data/pcb/good')
with open('models/phase1/simple_anomaly_detector.pkl', 'wb') as f:
    pickle.dump({'reference_stats': inspector.reference_stats, 'threshold': 0.1}, f)
"
```

### Dataset Guidelines

- **Image Quality**: 640x640 minimum, consistent lighting
- **Format**: JPEG/PNG, RGB color
- **Labels**: YOLO format, one .txt per image
- **Balance**: Similar number of samples per defect class
- **Splits**: 70% train, 20% val, 10% test recommended

📖 **Detailed Guide**: See [`ADD_NEW_DATASET.md`](ADD_NEW_DATASET.md) for comprehensive instructions.

---

### **1. Streamlit Web App (Recommended)**
```bash
streamlit run demo.py
```
- Interactive browser interface
- File upload and real-time results
- Visual defect annotations
- Anomaly heatmap display

### **2. FastAPI Server**
```bash
python src/local_server.py
```
- RESTful API for integration
- Automatic API documentation at `/docs`
- Production-ready with error handling

### **3. Basic Image Processing Demo**
```bash
python simple_demo.py
```
- No AI models required
- Demonstrates image processing pipeline
- Creates synthetic PCB for testing

---

## ⚙️ Configuration

### **Environment Variables**
```bash
# Model paths
PATCHCORE_PATH=./models/phase1/patchcore.onnx
YOLO_PATH=./models/phase2/yolov8m.onnx

# Detection threshold
PC_THRESHOLD=0.52

# Server settings
HOST=localhost
PORT=8000
```

### **YOLOv8 Configuration** (`configs/pcb.yaml`)
```yaml
path: ./data/pcb_labeled
train: images/train
val: images/val
test: images/test

nc: 5  # Number of classes

names:
  0: missing_component
  1: solder_bridge
  2: cold_joint
  3: trace_crack
  4: contamination
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/your-username/pcb-inspection.git
cd pcb-inspection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_local.py
```

### **Code Style**
- Follow PEP 8 Python style guide
- Use type hints for function parameters
- Add docstrings to all functions
- Write tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Project Maintainer:** [Your Name]

- **Email:** your.email@example.com
- **GitHub:** [@your-username](https://github.com/your-username)
- **LinkedIn:** [Your LinkedIn](https://linkedin.com/in/your-profile)

### **Support**
- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/your-username/pcb-inspection/issues)
- 💡 **Feature Requests:** [GitHub Discussions](https://github.com/your-username/pcb-inspection/discussions)
- 📧 **General Questions:** your.email@example.com

---

## 🙏 Acknowledgments

- **Anomalib** team for the PatchCore implementation
- **Ultralytics** team for YOLOv8
- **PyTorch** team for the deep learning framework
- **OpenCV** community for computer vision tools

---

<div align="center">

**Made with ❤️ for automated PCB quality control**

⭐ **Star this repository** if you find it helpful!

</div>
```
- REST API for integration
- Test with curl or Postman
- Minimal dependencies

---

## System Requirements

- **CPU**: Any modern CPU (i5/i7 or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and data
- **OS**: Windows, macOS, or Linux
- **Python**: 3.8+

**No GPU required** - all models run on CPU via ONNX Runtime.

---

## Training Tips

### Phase 1 (PatchCore)
- Use high-quality, defect-free PCB images
- 200-500 images recommended for good performance
- Images should be well-lit and in focus
- Vary angles and lighting conditions if possible

### Phase 2 (YOLOv8)
- Label defects using labelImg tool
- Aim for 200+ instances per defect class
- Include various defect severities
- Balance classes using the monitor script

### Performance Optimization
- Smaller batch sizes (16-32) for CPU training
- Use image size 256x256 for faster training
- Coreset ratio 0.01-0.05 for memory efficiency
```

**Response:**
```json
{
  "pass": false,
  "defects": [
    {
      "source": "yolov8",
      "class": "solder_bridge",
      "confidence": 0.87,
      "bbox": [120.5, 340.2, 195.1, 390.8]
    }
  ],
  "anomaly_score": 0.74,
  "latency_ms": 98.3
}
```

### `GET /metrics`
Returns aggregated stats from last 1000 inspections.

```bash
curl http://localhost:8000/metrics
```

### `GET /health`
Liveness check — returns `{"status": "ok"}`.

---

## Project Structure

```
pcb_inspection/
├── src/
│   ├── preprocessing.py      # PCBPreprocessor — align + CLAHE + augment
│   ├── train_phase1.py       # PatchCore training + ONNX export
│   ├── train_phase2.py       # YOLOv8 fine-tuning + ONNX export
│   ├── inspector.py          # PCBInspector + EnsembleInspector
│   ├── server.py             # FastAPI production server
│   ├── evaluate.py           # Full evaluation suite + escape rate
│   ├── monitor.py            # Class balance + drift detection
│   └── augment_defects.py    # Bbox-aware defect augmentation
├── configs/
│   └── pcb.yaml              # YOLO dataset config
├── data/                     # Your images go here
├── models/                   # Trained ONNX models go here
├── logs/                     # Inference logs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## KPI Targets

| KPI | Target | Measured By |
|-----|--------|-------------|
| Defect Escape Rate | < 0.1% | `evaluate.py` |
| False Positive Rate | < 5% | `evaluate.py` |
| Inference Latency P99 | < 200ms | `/metrics` endpoint |
| Phase 1 Image AUROC | > 0.95 | `train_phase1.py` output |
| Phase 2 mAP50 | > 0.85 | `train_phase2.py` output |
| System Uptime | > 99.5% | Grafana dashboard |

---

## Continuous Improvement

**Weekly:** Harvest hard negatives (score 0.4–0.6) for human review
```bash
python src/monitor.py --harvest
```

**Monthly:** Retrain both models with expanded dataset, validate before deploy

**Ongoing:** Monitor score drift
```bash
python src/monitor.py --drift-check --baseline 0.32
```

---

## 8-Week Timeline

| Week | Milestone |
|------|-----------|
| W1–2 | Camera rig + data setup |
| W2–3 | Preprocessing pipeline |
| W3–4 | Train + deploy Phase 1 |
| W4–5 | Deploy Phase 1, begin collecting defects |
| W5–6 | Label defects (200+ per class) |
| W6–8 | Train YOLOv8 Phase 2 |
| W8+  | Ensemble live + automated retraining |

---

## Camera Rig Tips

- **Lighting matters more than resolution** — use 6000K ring LED + 45° side-fill
- Consistent illumination prevents false positives from shadows
- 45° side-light reveals solder joint topology (cold joints appear dull vs. shiny)
- Use photoelectric trigger on conveyor for precise capture timing
- Fix focal length — no autofocus during production
