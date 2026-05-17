"""
PCB Preprocessing Pipeline
Handles image alignment, lighting normalization, and augmentation.
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path


class PCBPreprocessor:
    """
    Preprocesses raw PCB images for training and inference.

    Steps:
      1. Align PCB to remove rotation (contour-based)
      2. Normalize lighting with CLAHE
      3. Apply augmentations (train) or resize+normalize (inference)
    """

    def __init__(self, target_size=(640, 640)):
        self.size = target_size

        # Training augmentations — adds robustness to lighting/position variation
        self.train_aug = A.Compose([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
            A.GaussNoise(var_limit=(5, 25), p=0.3),
            A.Resize(*target_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

        # Inference augmentations — deterministic resize + normalize only
        self.infer_aug = A.Compose([
            A.Resize(*target_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

    def align_pcb(self, img: np.ndarray) -> np.ndarray:
        """
        Rotates the image so the PCB board is axis-aligned.
        Uses the largest contour (the board edge) to determine angle.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            return img  # No contours found — return as-is

        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        angle = rect[2]

        # Correct angle convention
        if abs(angle) > 45:
            angle = 90 + angle

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    def normalize_lighting(self, img: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        on the L-channel of LAB color space for uniform lighting.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def process(self, img_path: str, mode: str = "train") -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            img_path: Path to input image
            mode: "train" (with augmentation) or "infer" (deterministic)

        Returns:
            Preprocessed image tensor (H, W, C) normalized to ImageNet stats
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img = self.align_pcb(img)
        img = self.normalize_lighting(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug = self.train_aug if mode == "train" else self.infer_aug
        return aug(image=img)["image"]

    def process_from_array(self, img_bgr: np.ndarray,
                           mode: str = "infer") -> np.ndarray:
        """
        Process an already-loaded BGR image (e.g. from camera capture).
        """
        img = self.align_pcb(img_bgr)
        img = self.normalize_lighting(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.train_aug if mode == "train" else self.infer_aug
        return aug(image=img)["image"]


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path>")
        sys.exit(1)

    preprocessor = PCBPreprocessor()
    result = preprocessor.process(sys.argv[1], mode="infer")
    print(f"Output shape : {result.shape}")
    print(f"Output dtype : {result.dtype}")
    print(f"Value range  : [{result.min():.3f}, {result.max():.3f}]")
