"""
PCB Inspection: Phase 1 (PatchCore) + Phase 2 (YOLOv8) Ensemble
=================================================================
- PCBInspector    : ONNX-based PatchCore wrapper with heatmap generation
- EnsembleInspector: Combines both models — YOLOv8 classifies, PatchCore catches unknowns
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import onnxruntime as ort


# ── Phase 1: PatchCore ONNX Wrapper ──────────────────────────────────────────

class PCBInspector:
    """
    Runs PatchCore anomaly detection via ONNX Runtime.
    Returns anomaly score, pass/fail decision, and heatmap.
    """

    def __init__(self, onnx_path: str, threshold: float = 0.5):
        """
        Args:
            onnx_path : Path to patchcore.onnx
            threshold : Anomaly score cutoff — tune on validation set
        """
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],  # Force CPU for compatibility
        )
        self.threshold = threshold
        self.input_name = self.session.get_inputs()[0].name
        print(f"[PCBInspector] Loaded: {onnx_path}")
        print(f"[PCBInspector] Threshold: {threshold}")
        print(f"[PCBInspector] Provider: {self.session.get_providers()[0]}")

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize + normalize to ImageNet stats, return NCHW float32."""
        img = cv2.resize(img, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / \
              np.array([0.229, 0.224, 0.225])
        return img.transpose(2, 0, 1)[np.newaxis]  # NCHW

    def inspect(self, img_bgr: np.ndarray) -> dict:
        """
        Run anomaly detection on a single BGR image.

        Returns:
            {
              "defective": bool,
              "score": float (0-1, higher = more anomalous),
              "heatmap": np.ndarray (H, W) normalized 0-1
            }
        """
        inp = self.preprocess(img_bgr)
        anomaly_map, score = self.session.run(None, {self.input_name: inp})

        # Smooth and normalize heatmap
        hmap = gaussian_filter(anomaly_map[0, 0], sigma=4)
        hmap = (hmap - hmap.min()) / (hmap.max() + 1e-8)

        return {
            "defective": float(score[0]) > self.threshold,
            "score": float(score[0]),
            "heatmap": hmap,
        }

    def get_colored_heatmap(self, heatmap: np.ndarray,
                            original_img: np.ndarray) -> np.ndarray:
        """
        Overlay the anomaly heatmap on the original image for visualization.
        Red = high anomaly, Blue = normal.
        """
        h, w = original_img.shape[:2]
        hmap_resized = cv2.resize(heatmap, (w, h))
        hmap_uint8 = (hmap_resized * 255).astype(np.uint8)
        hmap_colored = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(original_img, 0.6, hmap_colored, 0.4, 0)


# ── Phase 2: YOLOv8 Wrapper ──────────────────────────────────────────────────

class YOLOInspector:
    """
    Runs YOLOv8 object detection for classified defect bounding boxes.
    """

    CLASS_NAMES = [
        "missing_component",
        "solder_bridge",
        "cold_joint",
        "trace_crack",
        "contamination",
    ]

    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        print(f"[YOLOInspector] Loaded: {model_path}")

    def inspect(self, img_bgr: np.ndarray) -> list[dict]:
        """
        Returns a list of detected defects with class + bounding box.
        """
        preds = self.model(img_bgr, conf=self.conf, iou=self.iou, verbose=False)
        defects = []
        for box in preds[0].boxes:
            defects.append({
                "source": "yolov8",
                "class": preds[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2] pixels
            })
        return defects


# ── Ensemble: Phase 1 + Phase 2 ──────────────────────────────────────────────

class EnsembleInspector:
    """
    Two-model ensemble:
    - YOLOv8 classifies known defect types with bounding boxes
    - PatchCore catches unknown anomalies that YOLOv8 hasn't been trained on

    Decision logic:
      FAIL if YOLOv8 detects any defect  OR
           PatchCore flags anomaly with no YOLOv8 detections (unknown anomaly)
    """

    def __init__(self,
                 patchcore_path: str,
                 yolo_path: str,
                 pc_threshold: float = 0.5,
                 yolo_conf: float = 0.4):
        self.phase1 = PCBInspector(patchcore_path, threshold=pc_threshold)
        self.phase2 = YOLOInspector(yolo_path, conf=yolo_conf)

    def inspect(self, img_bgr: np.ndarray) -> dict:
        """
        Full ensemble inspection.

        Returns:
            {
              "pass": bool,
              "defects": list of defect dicts,
              "anomaly_score": float,
              "heatmap": np.ndarray,
              "heatmap_overlay": np.ndarray  (visual overlay)
            }
        """
        pc_result = self.phase1.inspect(img_bgr)
        yolo_defects = self.phase2.inspect(img_bgr)

        defects = list(yolo_defects)  # Start with classified defects

        # PatchCore catches anomalies YOLOv8 hasn't seen before
        if pc_result["defective"] and not defects:
            defects.append({
                "source": "patchcore",
                "class": "unknown_anomaly",
                "confidence": pc_result["score"],
                "bbox": None,
            })

        return {
            "pass": len(defects) == 0,
            "defects": defects,
            "anomaly_score": pc_result["score"],
            "heatmap": pc_result["heatmap"],
            "heatmap_overlay": self.phase1.get_colored_heatmap(
                pc_result["heatmap"], img_bgr),
        }


# ── Threshold Tuning Utility ─────────────────────────────────────────────────

def find_optimal_threshold(scores: np.ndarray,
                           labels: np.ndarray) -> float:
    """
    Find the anomaly score threshold that maximises F1-score.

    Args:
        scores : Phase 1 anomaly scores for validation images
        labels : Ground truth (0=good, 1=defective)

    Returns:
        Optimal threshold float
    """
    from sklearn.metrics import roc_curve, f1_score

    fpr, tpr, thresholds = roc_curve(labels, scores)
    f1s = [f1_score(labels, scores > t) for t in thresholds]
    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]

    print(f"Optimal threshold : {best_t:.4f}")
    print(f"Best F1           : {max(f1s):.4f}")
    print(f"TPR at best F1    : {tpr[best_idx]:.4f}")
    print(f"FPR at best F1    : {fpr[best_idx]:.4f}")

    return best_t


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python inspector.py <patchcore.onnx> <yolov8m.onnx/pt>"
              " <image.jpg>")
        sys.exit(1)

    pc_path, yolo_path, img_path = sys.argv[1], sys.argv[2], sys.argv[3]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read: {img_path}")
        sys.exit(1)

    inspector = EnsembleInspector(pc_path, yolo_path)
    result = inspector.inspect(img)

    status = "✓ PASS" if result["pass"] else "✗ FAIL"
    print(f"\nResult        : {status}")
    print(f"Anomaly score : {result['anomaly_score']:.4f}")
    print(f"Defects found : {len(result['defects'])}")
    for d in result["defects"]:
        print(f"  [{d['source']}] {d['class']} — conf={d['confidence']:.3f}")

    # Save heatmap overlay
    cv2.imwrite("heatmap_overlay.jpg", result["heatmap_overlay"])
    print("\nHeatmap saved → heatmap_overlay.jpg")
