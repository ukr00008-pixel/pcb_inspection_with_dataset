"""
Simplified PCB Inspector — Basic Anomaly Detection (No Anomalib Required)
==========================================================================
Basic computer vision-based anomaly detection for PCB inspection.
Works with any Python version, no complex ML dependencies.
"""

import cv2
import numpy as np
from pathlib import Path
import os


class SimplePCBInspector:
    """
    Basic anomaly detection using traditional computer vision techniques.
    No machine learning required - uses statistical analysis and image processing.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Anomaly score cutoff (0-1, higher = more anomalous)
        """
        self.threshold = threshold
        self.reference_stats = None
        print(f"[SimplePCBInspector] Initialized with threshold: {threshold}")

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Basic preprocessing: resize and convert to grayscale."""
        # Resize to standard size
        resized = cv2.resize(img, (416, 416))

        # Convert to grayscale for analysis
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        return gray

    def extract_features(self, img: np.ndarray) -> dict:
        """Extract basic statistical features from the image."""
        features = {}

        # Basic statistics
        features['mean'] = np.mean(img)
        features['std'] = np.std(img)
        features['min'] = np.min(img)
        features['max'] = np.max(img)

        # Edge detection
        edges = cv2.Canny(img, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (img.shape[0] * img.shape[1])

        # Color analysis (if RGB)
        if len(img.shape) == 3:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            features['hue_mean'] = np.mean(hsv[:, :, 0])
            features['saturation_mean'] = np.mean(hsv[:, :, 1])
            features['value_mean'] = np.mean(hsv[:, :, 2])

        # Texture analysis using GLCM-like features
        features['contrast'] = self._calculate_contrast(img)
        features['homogeneity'] = self._calculate_homogeneity(img)

        return features

    def _calculate_contrast(self, img: np.ndarray) -> float:
        """Calculate contrast using local variance."""
        # Simple contrast measure using Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        return laplacian.var()

    def _calculate_homogeneity(self, img: np.ndarray) -> float:
        """Calculate homogeneity using inverse difference."""
        # Simple homogeneity measure
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = cv2.filter2D(img.astype(np.float32), -1, kernel)
        return 1.0 / (1.0 + np.mean(np.abs(filtered)))

    def train(self, good_images_dir: str):
        """Train on good PCB images to establish baseline statistics."""
        print(f"[SimplePCBInspector] Training on images from: {good_images_dir}")

        good_images_path = Path(good_images_dir)
        if not good_images_path.exists():
            raise FileNotFoundError(f"Good images directory not found: {good_images_dir}")

        features_list = []
        image_count = 0

        # Process all good images
        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        for ext in image_extensions:
            for img_path in good_images_path.glob(ext):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    processed = self.preprocess(img)
                    features = self.extract_features(processed)
                    features_list.append(features)
                    image_count += 1

                    if image_count % 10 == 0:
                        print(f"  Processed {image_count} images...")

                except Exception as e:
                    print(f"  Warning: Could not process {img_path}: {e}")
                    continue

        if not features_list:
            raise ValueError("No valid training images found")

        # Calculate reference statistics
        self.reference_stats = {}
        feature_names = features_list[0].keys()

        for feature_name in feature_names:
            values = [f[feature_name] for f in features_list]
            self.reference_stats[feature_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        print(f"[SimplePCBInspector] Training complete. Processed {image_count} images.")
        print(f"[SimplePCBInspector] Learned {len(feature_names)} features.")

        return self.reference_stats

    def inspect(self, img_bgr: np.ndarray) -> dict:
        """
        Inspect a single PCB image for anomalies.

        Returns:
            {
              "defective": bool,
              "score": float (0-1, higher = more anomalous),
              "feature_scores": dict of individual feature anomalies
            }
        """
        if self.reference_stats is None:
            raise ValueError("Inspector not trained. Call train() first.")

        # Preprocess image
        processed = self.preprocess(img_bgr)
        features = self.extract_features(processed)

        # Calculate anomaly scores for each feature
        feature_scores = {}
        total_score = 0

        for feature_name, feature_value in features.items():
            if feature_name in self.reference_stats:
                ref = self.reference_stats[feature_name]
                # Calculate z-score (how many standard deviations from mean)
                if ref['std'] > 0:
                    z_score = abs(feature_value - ref['mean']) / ref['std']
                    # Normalize to 0-1 scale (z-score of 3 = very anomalous)
                    feature_score = min(z_score / 3.0, 1.0)
                else:
                    # If no variation in training data, any difference is anomalous
                    feature_score = 1.0 if feature_value != ref['mean'] else 0.0

                feature_scores[feature_name] = feature_score
                total_score += feature_score

        # Average anomaly score across all features
        avg_score = total_score / len(features)

        # Create simple heatmap (just edge density for visualization)
        edges = cv2.Canny(processed, 50, 150)
        heatmap = cv2.normalize(edges.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

        return {
            "defective": avg_score > self.threshold,
            "score": avg_score,
            "feature_scores": feature_scores,
            "heatmap": heatmap
        }

    def get_colored_heatmap(self, heatmap: np.ndarray, original_img: np.ndarray) -> np.ndarray:
        """Create colored heatmap overlay on original image."""
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)


class YOLOInspector:
    """
    YOLOv8 wrapper for defect classification.
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
        print(f"[YOLOInspector] Loaded model: {model_path}")

    def inspect(self, img: np.ndarray) -> list:
        """Run YOLOv8 inference and return detected defects."""
        results = self.model(img, conf=self.conf, iou=self.iou)

        defects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                defects.append({
                    'class': self.CLASS_NAMES[cls] if cls < len(self.CLASS_NAMES) else f'class_{cls}',
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

        return defects


class EnsembleInspector:
    """
    Combines simple anomaly detection with YOLOv8 classification.
    """

    def __init__(self, patchcore_path: str = None, yolo_path: str = None, pc_threshold: float = 0.5, patchcore_inspector=None):
        self.patchcore = patchcore_inspector  # Use provided inspector if available
        self.yolo = None
        self.pc_threshold = pc_threshold

        # If no inspector provided but path given, try to load it
        if self.patchcore is None and patchcore_path and Path(patchcore_path).exists():
            try:
                import pickle
                with open(patchcore_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.patchcore = SimplePCBInspector(threshold=model_data['threshold'])
                self.patchcore.reference_stats = model_data['reference_stats']
                print(f"[EnsembleInspector] Loaded PatchCore from: {patchcore_path}")
            except Exception as e:
                print(f"[EnsembleInspector] Failed to load PatchCore: {e}")

        if yolo_path and Path(yolo_path).exists():
            self.yolo = YOLOInspector(yolo_path)
        else:
            print("[EnsembleInspector] No YOLO model provided - classification disabled")

    def inspect(self, img: np.ndarray) -> dict:
        """Run ensemble inspection."""
        result = {
            "pass": True,
            "defects": [],
            "anomaly_score": 0.0,
            "heatmap": None
        }

        # Run anomaly detection if available
        if self.patchcore:
            try:
                anomaly_result = self.patchcore.inspect(img)
                result["anomaly_score"] = anomaly_result["score"]
                result["heatmap"] = anomaly_result["heatmap"]

                # If anomaly detected, mark as potential defect
                if anomaly_result["defective"]:
                    result["pass"] = False
                    result["defects"].append({
                        "class": "anomaly",
                        "confidence": min(anomaly_result["score"], 1.0),
                        "bbox": [0, 0, img.shape[1], img.shape[0]]  # Full image
                    })
            except Exception as e:
                print(f"[EnsembleInspector] Anomaly detection failed: {e}")

        # Run defect classification if available
        if self.yolo:
            try:
                defects = self.yolo.inspect(img)
                result["defects"].extend(defects)

                # If any defects found, mark as fail
                if defects:
                    result["pass"] = False
            except Exception as e:
                print(f"[EnsembleInspector] Defect classification failed: {e}")

        return result