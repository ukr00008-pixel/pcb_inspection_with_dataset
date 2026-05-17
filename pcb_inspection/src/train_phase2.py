"""
Phase 2 Training: YOLOv8 Supervised Defect Detection
=====================================================
Fine-tunes YOLOv8-m on labeled PCB defect images.
Requires 200+ labeled examples per defect class.

Usage:
    python src/train_phase2.py
    python src/train_phase2.py --epochs 200 --batch 8
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on PCB defect dataset")
    parser.add_argument("--config", default="configs/pcb.yaml",
                        help="YOLO dataset YAML config")
    parser.add_argument("--model", default="yolov8m.pt",
                        choices=["yolov8s.pt", "yolov8m.pt", "yolov8x.pt"],
                        help="YOLOv8 model size: s=fast, m=balanced, x=accurate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (fewer for CPU)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (smaller for CPU)")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Image size (smaller = faster)")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu for CPU training")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Number of backbone layers to freeze (head-only training)")
    parser.add_argument("--output", default="./models/phase2",
                        help="Where to copy final ONNX model")
    return parser.parse_args()


def train(args):
    from ultralytics import YOLO
    import shutil

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 2] Config  : {args.config}")
    print(f"[Phase 2] Model   : {args.model}")
    print(f"[Phase 2] Epochs  : {args.epochs}")
    print(f"[Phase 2] Batch   : {args.batch}")
    print(f"[Phase 2] Freeze  : first {args.freeze} backbone layers")

    # ── Load pretrained YOLOv8 weights ───────────────────────────────────────
    model = YOLO(args.model)  # Downloads from Ultralytics if not cached

    # ── Fine-tune on PCB defects ──────────────────────────────────────────────
    results = model.train(
        data=args.config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=30,              # Early stopping if no improvement for 30 epochs
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=5,
        weight_decay=0.0005,

        # ── Augmentations ─────────────────────────────────────────────────────
        mosaic=1.0,               # Mosaic 4-image augmentation
        mixup=0.1,                # Mix two images for generalization
        degrees=15,               # Random rotation ±15°
        fliplr=0.5,               # Horizontal flip
        hsv_s=0.7,                # Hue-Saturation-Value shift

        # ── Transfer learning: freeze backbone, train detection head first ────
        freeze=args.freeze,

        project="runs/pcb_detect",
        name="yolov8m_phase2",
        exist_ok=True,
    )

    # ── Evaluate on validation set ────────────────────────────────────────────
    print("\n[Phase 2] Evaluating on validation set...")
    metrics = model.val()

    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    print(f"\n{'='*50}")
    print(f"  mAP50     : {map50:.4f}   (target > 0.85)")
    print(f"  mAP50-95  : {map50_95:.4f}  (target > 0.60)")
    print(f"{'='*50}")

    if map50 < 0.85:
        print("  ⚠ mAP50 below target — collect more labeled defects")
        print("  Tip: Run check_class_balance to check for imbalanced classes")
    else:
        print("  ✓ mAP50 target achieved!")

    # ── Export to ONNX ────────────────────────────────────────────────────────
    print(f"\n[Phase 2] Exporting ONNX → {output_dir}/yolov8m.onnx")
    model.export(format="onnx", opset=17, simplify=True, imgsz=args.imgsz)

    # Copy best ONNX to models/phase2/
    run_dir = Path("runs/pcb_detect/yolov8m_phase2")
    onnx_src = run_dir / "weights" / "best.onnx"
    if onnx_src.exists():
        shutil.copy(onnx_src, output_dir / "yolov8m.onnx")
        print(f"[Phase 2] Saved → {output_dir}/yolov8m.onnx")
    else:
        print(f"[Phase 2] ⚠ ONNX file not found at expected path: {onnx_src}")

    return {"map50": map50, "map50_95": map50_95}


if __name__ == "__main__":
    args = parse_args()
    train(args)
