# 🎯 YOLOv8 Training - Quick Start

## ✅ What's Happening Now
Your YOLOv8 model is **training on 350 labeled PCB images** to detect defects:
- cold_joint
- contamination  
- missing_component
- solder_bridge
- trace_crack

**Training Time**: ~15-25 minutes on CPU

---

## ⏳ WHILE TRAINING IS RUNNING

In a **new terminal**, run the auto-deployment script:

```bash
cd d:\pcb_inspection_with_dataset\pcb_inspection
python auto_deploy_model.py
```

This will:
1. Wait for training to complete
2. Automatically move the model to the right location
3. Tell you when to restart the server

---

## ✅ WHEN TRAINING FINISHES

The `auto_deploy_model.py` script will notify you.

Then:
1. **Stop the current server** (Ctrl+C in terminal running `local_server.py`)
2. **Restart the server**:
   ```bash  
   python src/local_server.py
   ```
3. **Test it**:
   ```bash
   python test_api.py
   ```

---

## 📊 Expected Results After Training

Your API will now detect defects:

```json
POST /inspect with defective_pcb.jpg

Response:
{
  "pass": false,
  "defects": [
    {
      "class": "cold_joint",
      "confidence": 0.92,
      "bbox": [120, 200, 180, 240]
    }
  ],
  "anomaly_score": 0.35,
  "latency_ms": 12.5
}
```

---

## 📚 Full Documentation

Read the detailed guide:
- **File**: `YOLO_TRAINING_GUIDE.md`
- **Location**: `pcb_inspection/YOLO_TRAINING_GUIDE.md`

---

## 🆘 Need Help?

### Training Still Running?
- Don't close the terminal
- `auto_deploy_model.py` will wait automatically
- Check system temperature if it's slow

### Want to Train Faster?
- Use GPU (if available): Edit `train_yolo_simple.py` line with `device="cuda"`
- Or wait for CPU training ~20 mins

### Ready to Start Over?
```bash
# Delete training outputs
rmdir /s ./models/phase2
rmdir /s ./runs

# Retrain
python train_yolo_simple.py
```

---

**Current Status**: ⏳ Training (30 epochs on CPU)
**Next Action**: Run `python auto_deploy_model.py` in another terminal
**Estimated Completion**: 15-25 minutes from start
