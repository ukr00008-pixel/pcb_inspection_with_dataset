"""
Defect Augmentation Pipeline
=============================
Multiplies labeled defect images using bbox-aware augmentations.
Use this to expand a small defect dataset before Phase 2 training.

Usage:
    python src/augment_defects.py \\
        --images data/pcb_labeled/images/train \\
        --labels data/pcb_labeled/labels/train \\
        --output data/pcb_labeled_aug \\
        --multiplier 5
"""

import argparse
import cv2
import numpy as np
import shutil
from pathlib import Path
import albumentations as A


# ── Bbox-aware augmentation pipeline ─────────────────────────────────────────

DEFECT_AUG = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.HueSaturationValue(10, 20, 10, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.4),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.CoarseDropout(num_holes_range=(1, 4),
                    hole_height_range=(10, 20),
                    hole_width_range=(10, 20), p=0.3),
], bbox_params=A.BboxParams(
    format="yolo",
    label_fields=["class_labels"],
    min_visibility=0.3,      # Drop box if < 30% visible after augmentation
))


def read_yolo_labels(label_path: str) -> tuple[list, list]:
    """Read YOLO label file → (class_ids, bboxes_xywh_normalized)."""
    class_ids, bboxes = [], []
    text = Path(label_path).read_text().strip()
    for line in text.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        class_ids.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:5]])
    return class_ids, bboxes


def write_yolo_labels(label_path: str, class_ids: list, bboxes: list):
    """Write YOLO label file from class_ids and bboxes."""
    lines = []
    for cls, box in zip(class_ids, bboxes):
        lines.append(f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
    Path(label_path).write_text("\n".join(lines))


def augment_dataset(images_dir: str,
                    labels_dir: str,
                    output_dir: str,
                    multiplier: int = 5):
    """
    Augments each (image, label) pair `multiplier` times.
    Copies originals + writes augmented versions to output_dir.
    """
    img_dir = Path(images_dir)
    lbl_dir = Path(labels_dir)
    out_img_dir = Path(output_dir) / "images"
    out_lbl_dir = Path(output_dir) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    total_generated = 0

    for img_path in image_files:
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"  ⚠ No label for {img_path.name} — skipping")
            continue

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_ids, bboxes = read_yolo_labels(str(label_path))

        if not bboxes:
            continue

        # ── Copy original ──────────────────────────────────────────────────
        shutil.copy(img_path, out_img_dir / img_path.name)
        shutil.copy(label_path, out_lbl_dir / label_path.name)

        # ── Generate augmented versions ────────────────────────────────────
        for i in range(multiplier):
            aug_result = DEFECT_AUG(
                image=img_rgb,
                bboxes=bboxes,
                class_labels=class_ids,
            )

            aug_img = cv2.cvtColor(aug_result["image"], cv2.COLOR_RGB2BGR)
            aug_boxes = aug_result["bboxes"]
            aug_classes = aug_result["class_labels"]

            if not aug_boxes:
                continue  # All boxes cropped out — skip this augmentation

            stem = f"{img_path.stem}_aug{i:03d}"
            cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), aug_img)
            write_yolo_labels(
                str(out_lbl_dir / f"{stem}.txt"), aug_classes, aug_boxes)
            total_generated += 1

    print(f"\nAugmentation complete!")
    print(f"  Original images : {len(image_files)}")
    print(f"  Generated       : {total_generated}")
    print(f"  Output          : {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Defect Augmentation")
    parser.add_argument("--images",     required=True)
    parser.add_argument("--labels",     required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--multiplier", type=int, default=5,
                        help="Number of augmented copies per image (default: 5)")
    args = parser.parse_args()

    augment_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        multiplier=args.multiplier,
    )
