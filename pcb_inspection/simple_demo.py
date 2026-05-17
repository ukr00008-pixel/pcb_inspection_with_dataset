"""
Simple PCB Demo — No Models Required
====================================
Basic demonstration of image processing pipeline.

Usage:
    python simple_demo.py
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def demonstrate_image_processing():
    """Show basic image processing capabilities."""
    print("🔍 PCB Inspection - Basic Image Processing Demo")
    print("=" * 50)

    # Create a synthetic PCB-like image for demonstration
    img = create_synthetic_pcb()

    print("✅ Created synthetic PCB image")
    print(f"   Image size: {img.shape}")
    print(f"   Data type: {img.dtype}")

    # Demonstrate preprocessing
    processed = preprocess_image(img)
    print("✅ Applied preprocessing (resize, normalize)")

    # Show some basic analysis
    analysis = analyze_image(processed)
    print("✅ Performed basic image analysis")
    print(f"   Brightness: {analysis['brightness']:.2f}")
    print(f"   Contrast: {analysis['contrast']:.2f}")
    print(f"   Sharpness: {analysis['sharpness']:.2f}")

    # Save demo image
    output_path = Path("./demo_output.png")
    cv2.imwrite(str(output_path), img)
    print(f"✅ Saved demo image to {output_path}")

    print("\n🎉 Demo complete!")
    print("This shows the basic image processing pipeline.")
    print("To run the full AI inspection, train models first.")

def create_synthetic_pcb():
    """Create a synthetic PCB-like image for demonstration."""
    # Create a green PCB-like background
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[:, :] = [20, 60, 20]  # Dark green background

    # Add some copper traces (brown lines)
    for i in range(10):
        y = 100 + i * 50
        cv2.line(img, (50, y), (590, y), (30, 60, 100), 8)

    # Add some components (rectangles)
    components = [
        (150, 150, 50, 50, (80, 80, 80)),  # IC
        (300, 200, 40, 60, (60, 60, 60)),  # Capacitor
        (450, 300, 35, 35, (100, 100, 100)),  # Resistor
    ]

    for x, y, w, h, color in components:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)

    # Add some text
    cv2.putText(img, "DEMO PCB", (250, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    return img

def preprocess_image(img):
    """Basic preprocessing: resize and normalize."""
    # Resize to standard size
    resized = cv2.resize(img, (416, 416))

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    return normalized

def analyze_image(img):
    """Perform basic image analysis."""
    # Brightness (mean pixel value)
    brightness = np.mean(img)

    # Contrast (standard deviation)
    contrast = np.std(img)

    # Sharpness (using Laplacian variance)
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()

    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness
    }

if __name__ == "__main__":
    demonstrate_image_processing()