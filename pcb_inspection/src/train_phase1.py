"""
Phase 1 Training: PatchCore (Unsupervised Anomaly Detection)
============================================================
Trains on good PCB images only — no defect labels required.
Exports trained model to ONNX for production inference.

Usage:
    python src/train_phase1.py
    python src/train_phase1.py --data ./data/pcb --output ./models/phase1
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PatchCore anomaly detection model")
    parser.add_argument("--data", default="./data/pcb",
                        help="Root data dir containing good/ subfolder")
    parser.add_argument("--output", default="./models/phase1",
                        help="Directory to save checkpoints and ONNX model")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Input image size (smaller = faster training)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (smaller = less memory)")
    parser.add_argument("--coreset-ratio", type=float, default=0.01,
                        help="Coreset sampling ratio (1%% = fast + memory-efficient)")
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--backbone", default="resnet18",
                        choices=["wide_resnet50_2", "resnet18", "resnet50"],
                        help="Model backbone (resnet18 = fastest)")
    return parser.parse_args()


def train(args):
    # ── Imports ───────────────────────────────────────────────────────────────
    from anomalib.models import Patchcore
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.deploy import ExportType
    from lightning.pytorch.callbacks import ModelCheckpoint

    data_root = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 1] Data root  : {data_root}")
    print(f"[Phase 1] Output dir : {output_dir}")
    print(f"[Phase 1] Backbone   : {args.backbone}")
    print(f"[Phase 1] Image size : {args.image_size}x{args.image_size}")
    print(f"[Phase 1] Coreset    : {args.coreset_ratio*100:.1f}%")

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    datamodule = Folder(
        name="pcb_inspection",
        root=data_root,
        normal_dir="good",  # Simplified path for local data
        abnormal_dir="defects",
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=0,  # CPU-only, no multiprocessing
    )

    # ── 2. Model ──────────────────────────────────────────────────────────────
    pre_processor = Patchcore.configure_pre_processor(
        image_size=(args.image_size, args.image_size)
    )

    model = Patchcore(
        backbone=args.backbone,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        pre_processor=pre_processor,
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    engine = Engine(
        max_epochs=1,
        accelerator="cpu",  # Force CPU training
        devices=1,
        default_root_dir=str(output_dir),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(output_dir),
                filename="patchcore-pcb",
                monitor="pixel_AUROC",
                mode="max",
            ),
        ],
    )
    print("\n[Phase 1] Starting training...")
    engine.fit(model=model, datamodule=datamodule)
    print("\n[Phase 1] Evaluating on validation set...")
    metrics = engine.test(model=model, datamodule=datamodule)

    image_auroc = metrics[0].get("image_AUROC", 0)
    pixel_auroc = metrics[0].get("pixel_AUROC", 0)
    print(f"\n{'='*50}")
    print(f"  Image AUROC : {image_auroc:.4f}  (target > 0.95)")
    print(f"  Pixel AUROC : {pixel_auroc:.4f}  (target > 0.90)")
    print(f"{'='*50}")
    if image_auroc < 0.95:
        print("  ⚠ Image AUROC below target — collect more good images")
    else:
        print("  ✓ Image AUROC target achieved!")

        # ── 5. Export ONNX ────────────────────────────────────────────────────────
        onnx_path = output_dir / "patchcore.onnx"
        print(f"\n[Phase 1] Exporting ONNX → {onnx_path}")
        engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=str(output_dir),
        )
        print("[Phase 1] Export complete — ready for production deployment!")

    return {"image_auroc": image_auroc, "pixel_auroc": pixel_auroc}

if __name__ == "__main__":
    args = parse_args()
    train(args)