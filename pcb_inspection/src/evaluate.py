"""
Full Evaluation Suite
=====================
Evaluates the ensemble inspector on a labeled test set.
Reports: classification metrics, ROC-AUC, confusion matrix, defect escape rate.

Usage:
    python src/evaluate.py \\
        --phase1 models/phase1/patchcore.onnx \\
        --phase2 models/phase2/yolov8m.onnx \\
        --test-dir data/pcb_labeled/images/test \\
        --labels-dir data/pcb_labeled/labels/test
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PCB inspection ensemble")
    parser.add_argument("--phase1", required=True,
                        help="Path to patchcore.onnx")
    parser.add_argument("--phase2", required=True,
                        help="Path to yolov8m.onnx or .pt")
    parser.add_argument("--test-dir", required=True,
                        help="Directory of test images")
    parser.add_argument("--labels-dir", default=None,
                        help="Directory of YOLO label .txt files (optional)")
    parser.add_argument("--threshold", type=float, default=0.52,
                        help="PatchCore anomaly score threshold")
    parser.add_argument("--good-subdir", default="good",
                        help="Subfolder name for good-board images if structured")
    return parser.parse_args()


def load_test_set(test_dir: str):
    """
    Loads test images and infers labels from directory structure.
    Expects:  test_dir/good/*.jpg  and  test_dir/defects/*.jpg
    OR:       test_dir/*.jpg with label 0 assumed for all (good only).
    """
    test_path = Path(test_dir)
    images, labels = [], []

    good_dir = test_path / "good"
    defect_dir = test_path / "defects"

    if good_dir.exists():
        for p in sorted(good_dir.glob("*.jpg")) + sorted(good_dir.glob("*.png")):
            images.append(str(p))
            labels.append(0)  # 0 = good
    if defect_dir.exists():
        for p in sorted(defect_dir.glob("*.jpg")) + sorted(defect_dir.glob("*.png")):
            images.append(str(p))
            labels.append(1)  # 1 = defective
    if not images:
        # Flat directory — assume all are good (anomaly detection only)
        for p in sorted(test_path.glob("*.jpg")) + sorted(test_path.glob("*.png")):
            images.append(str(p))
            labels.append(0)

    return images, labels


def run_evaluation(inspector, test_images, test_labels):
    scores, preds, truths = [], [], []

    print(f"[Evaluate] Running on {len(test_images)} test images...")
    for i, (img_path, label) in enumerate(zip(test_images, test_labels)):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ Cannot read: {img_path}")
            continue

        result = inspector.inspect(img)
        scores.append(result["anomaly_score"])
        preds.append(0 if result["pass"] else 1)
        truths.append(label)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_images)} processed...")

    truths = np.array(truths)
    preds  = np.array(preds)
    scores = np.array(scores)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(truths, preds,
                                target_names=["PASS (good)", "FAIL (defect)"]))

    print("=" * 60)
    print("  KEY METRICS")
    print("=" * 60)

    if len(np.unique(truths)) > 1:
        auroc = roc_auc_score(truths, scores)
        ap    = average_precision_score(truths, scores)
        print(f"  ROC-AUC              : {auroc:.4f}  (target > 0.95)")
        print(f"  Average Precision    : {ap:.4f}")
    else:
        print("  (Only one class present — ROC-AUC skipped)")

    cm = confusion_matrix(truths, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        escape_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
        fpr         = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0

        print(f"\n  True Positives (defects caught)  : {tp}")
        print(f"  False Positives (good→rejected)  : {fp}")
        print(f"  True Negatives (good→passed)     : {tn}")
        print(f"  False Negatives (defects missed) : {fn}  ← CRITICAL")
        print(f"\n  Defect Escape Rate : {escape_rate:.2f}%   (target < 0.1%)")
        print(f"  False Positive Rate: {fpr:.2f}%   (target < 5%)")

        if escape_rate > 0.1:
            print("\n  ⚠ ESCAPE RATE EXCEEDS TARGET — lower threshold or collect"
                  " more defect examples")
        else:
            print("\n  ✓ Escape rate within target!")

    print("=" * 60)

    return {
        "scores": scores,
        "preds": preds,
        "truths": truths,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from inspector import EnsembleInspector

    args = parse_args()

    inspector = EnsembleInspector(
        patchcore_path=args.phase1,
        yolo_path=args.phase2,
        pc_threshold=args.threshold,
    )

    test_images, test_labels = load_test_set(args.test_dir)

    if not test_images:
        print(f"No images found in: {args.test_dir}")
        sys.exit(1)

    print(f"[Evaluate] Test set: {len(test_images)} images "
          f"({sum(test_labels)} defective, "
          f"{len(test_labels)-sum(test_labels)} good)")

    run_evaluation(inspector, test_images, test_labels)
