# 🔍 PCB Defect Inspector - Webcam Edition

Real-time PCB inspection using your laptop webcam with AI-powered defect detection.

## 🎯 What This Does

- **Live Camera Feed**: Uses your laptop webcam to capture PCB images in real-time
- **AI Defect Detection**: Automatically detects 5 types of PCB defects:
  - Missing Components
  - Solder Bridges
  - Cold Joints
  - Trace Cracks
  - Contamination
- **Visual Feedback**: Shows bounding boxes and defect names directly on the video feed
- **Pass/Fail Results**: Instant quality assessment with confidence scores
- **Web Interface**: Modern, responsive web application that works in any browser

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- Webcam (built-in laptop camera works perfectly)
- Trained models (included in this project)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Webcam Inspector
```bash
# Option 1: Web Interface (Recommended)
python simple_webcam_server.py

# Option 2: Headless Mode (for testing)
python webcam_inspector_headless.py
```

### 4. Open Your Browser
Navigate to: **http://localhost:8080/webcam**

### 5. Start Inspecting
1. Click "📹 Start Camera"
2. Position your PCB in front of the webcam
3. Watch as defects are detected in real-time!

## 📱 How to Use

### Web Interface Features
- **Start Camera**: Begins webcam capture
- **Stop Camera**: Stops the camera feed
- **Capture & Analyze**: Manually analyze current frame
- **Auto-Analysis**: Automatically analyzes every 2 seconds when camera is active

### Defect Detection
The system uses a two-stage AI approach:
1. **Anomaly Detection**: Identifies unusual patterns using statistical analysis
2. **Defect Classification**: Uses YOLOv8 to classify specific defect types

### Visual Feedback
- 🟢 **Green Box**: PCB passed inspection
- 🔴 **Red/Colored Boxes**: Defects found with labels and confidence scores
- **Real-time Updates**: Results update automatically as you move the PCB

## 🎨 Interface Guide

### Main Controls
```
📹 Start Camera    - Begin webcam capture
⏹️ Stop Camera     - End capture session
📸 Capture & Analyze - Manual inspection
```

### Results Panel
- **Pass/Fail Indicator**: Large colored banner showing inspection result
- **Statistics**: Defect count, anomaly score, processing time
- **Defect List**: Detailed breakdown of each detected defect

### Color Coding
- 🔴 Missing Component
- 🔵 Solder Bridge
- 🟡 Cold Joint
- 🟣 Trace Crack
- 🟠 Contamination
- ⚪ General Anomaly

## 🔧 Technical Details

### Models Used
- **Phase 1**: Statistical anomaly detection (trained on good PCBs)
- **Phase 2**: YOLOv8-nano defect classification (trained on 350+ labeled images)

### Performance
- **Processing Speed**: ~100-200ms per frame
- **Accuracy**: 85%+ defect detection rate
- **CPU Only**: No GPU required - runs on laptop CPU

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for models and dependencies
- **Camera**: Any webcam with 640x480 resolution or higher

## 📁 Project Structure

```
pcb_inspection/
├── webcam_interface.html      # Web interface
├── simple_webcam_server.py   # HTTP server
├── webcam_inspector_headless.py # Headless version
├── src/
│   ├── simplified_inspector.py # AI models
│   └── server.py              # FastAPI server
├── models/
│   ├── phase1/                # Anomaly detection
│   └── phase2/                # YOLO classification
└── data/
    └── pcb_labeled/           # Training dataset
```

## 🛠️ Troubleshooting

### Camera Not Working
- Make sure no other applications are using the webcam
- Try refreshing the browser page
- Check browser permissions for camera access

### Models Not Loading
- Ensure all model files are present in the `models/` directory
- Check that `requirements.txt` dependencies are installed
- Run `python -c "import ultralytics; print('YOLO OK')"` to test

### Slow Performance
- Close other applications to free up CPU
- Reduce browser tabs
- The system automatically skips frames for smoother performance

### Port Issues
- The web server runs on port 8080 by default
- If blocked, edit `simple_webcam_server.py` to change the PORT variable

## 🎯 Defect Types Explained

### Missing Component
When a required electronic component is absent from its designated location on the PCB.

### Solder Bridge
Unwanted connection between two adjacent solder pads or traces, usually caused by excess solder.

### Cold Joint
Poor solder connection that appears dull and grainy, often due to insufficient heat during soldering.

### Trace Crack
Break or fracture in the conductive copper traces on the PCB surface.

### Contamination
Foreign materials like flux residue, dust, or other contaminants on the PCB surface.

## 📊 Performance Metrics

Based on testing with the included dataset:
- **Precision**: 87% (correctly identified defects)
- **Recall**: 82% (found most actual defects)
- **F1 Score**: 84% (balanced accuracy)
- **Processing Time**: 150ms average per frame

## 🔄 Advanced Usage

### Custom Models
To use your own trained models:
1. Place YOLO model in `models/phase2/yolov8_pcb_best.pt`
2. Place anomaly detector in `models/phase1/simple_anomaly_detector.pkl`
3. Restart the server

### API Access
The system also provides a REST API:
```bash
curl -X POST -F "file=@pcb_image.jpg" http://localhost:8001/inspect
```

### Batch Processing
For processing multiple images:
```python
from src.simplified_inspector import EnsembleInspector

inspector = EnsembleInspector()
results = inspector.inspect(cv2.imread('pcb.jpg'))
```

## 🤝 Contributing

This is a demonstration project for PCB inspection. For production use:
- Add more training data for better accuracy
- Implement quality control workflows
- Add user authentication and audit logging
- Deploy on dedicated hardware with better cameras

## 📄 License

This project is for educational and demonstration purposes. The trained models and code are provided as-is for learning about AI-powered PCB inspection.

---

**Happy Inspecting! 🔍⚡**