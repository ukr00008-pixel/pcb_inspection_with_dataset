# YOLOv8 PCB Defect Detection - Training & Deployment Guide

## 📊 Current Status

Your YOLOv8 model is **currently training** on the following configuration:

```
Model:        YOLOv8-nano (ultra-lightweight)
Dataset:      350 labeled PCB images (5 defect types)
Epochs:       30 (for quick training on CPU)
Batch Size:   8 (CPU-friendly)
Device:       CPU (Intel Core i3-N305)
Expected Time: 15-25 minutes
```

### Defect Classes
- `0` - missing_component
- `1` - solder_bridge  
- `2` - cold_joint
- `3` - trace_crack
- `4` - contamination

## 🚀 Training Process

### Option 1: Monitor Training (Recommended)
```bash
# In a new terminal, watch training progress
python auto_deploy_model.py
```

This script will:
1. Automatically detect when training completes
2. Deploy the trained model to the API
3. Notify you when ready

### Option 2: Manual Deployment
Once training finishes (you'll see "✅ Training COMPLETE!" message):

```bash
# Find and deploy the model
python auto_deploy_model.py
```

## 🔄 Restart Server
After model deployment:
```bash
# Stop current server (Ctrl+C)
# Then restart:
python src/local_server.py
```

## 🧪 Test the API with YOLO

Once server restarts:
```bash
# Test with defective image
python test_api.py

# Expected response with detections:
{
  "pass": false,
  "defects": [
    {
      "class": "cold_joint",
      "confidence": 0.87,
      "bbox": [100, 150, 200, 220]
    }
  ],
  "anomaly_score": 0.45,
  "latency_ms": 15.2
}
```

## 📁 Model Location
- **Training output**: `./runs/detect/models/phase2/yolov8_pcb/weights/best.pt`
- **Deployment location**: `./models/phase2/yolov8_pcb_best.pt` (auto-copied by deploy script)

## 🎯 Ensemble Detection
Your API now runs **two-stage inspection**:

1. **Phase 1 (Statistical)**: Simple anomaly detection (~0.3ms)
2. **Phase 2 (YOLOv8)**: Precise defect classification (~10-15ms)

Combined inference: **~15-20ms per image on CPU**

## 💡 Customization

### Adjust Training Parameters
Edit `train_yolo_simple.py`:
```python
epochs=30,        # Increase for better accuracy
batch=8,          # Reduce if out of memory
imgsz=416,        # Image size (smaller = faster)
```

### Download Pretrained Weights
For future reference, models are cached locally:
- `~/.cache/yolov8/`
- Next training will be faster

## 📊 Performance Expectations

| Metric | Estimate |
|--------|----------|
| Inference Speed | 10-15ms per image (CPU) |
| Model Size | ~3.25 MB |
| Accuracy | Good (depends on training quality) |
| GPU Time | 5-10 minutes |
| CPU Time | 15-25 minutes |

## 🔧 Troubleshooting

### Training Stuck?
```bash
# Check available memory
# Reduce batch size to 4 in train_yolo_simple.py
```

### Import Errors?
```bash
pip install ultralytics torch torchvision
```

### Model Not Deploying?
```bash
# Check if training truly completed
ls -la ./models/phase2/
ls -la ./runs/detect/

# Manual deployment
cp ./runs/detect/models/phase2/yolov8_pcb/weights/best.pt \
   ./models/phase2/yolov8_pcb_best.pt
```

## 🎓 Next Steps

1. ⏳ **Let training complete** (mono-monitor with `auto_deploy_model.py`)
2. ✅ **Restart the server** once deployment is done
3. 🧪 **Test with real PCB images**
4. 📈 **Fine-tune hyperparameters** if needed
5. 🚀 **Deploy your completed system**

---

**Training started**: YOLOv8-nano is learning to detect PCB defects!
**Status**: In Progress... Check back soon! 🤖
