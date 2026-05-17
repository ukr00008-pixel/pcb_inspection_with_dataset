"""
Test the webcam capture and analyze functionality
"""
import requests
import cv2
import numpy as np
from pathlib import Path

def test_capture_analyze():
    """Test the capture and analyze functionality."""
    print("🧪 Testing capture and analyze functionality...")

    # Load a test PCB image
    test_image_path = Path("./data/pcb/good/01.jpg")  # Use a good PCB image for testing

    if not test_image_path.exists():
        print("❌ Test image not found. Looking for alternative...")
        # Try to find any image in the data directory
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images = list(Path("./data").glob(f"**/{ext}"))
            if images:
                test_image_path = images[0]
                break

    if not test_image_path.exists():
        print("❌ No test images found in data directory")
        return

    print(f"📸 Using test image: {test_image_path}")

    # Read and encode image
    img = cv2.imread(str(test_image_path))
    if img is None:
        print("❌ Could not load test image")
        return

    # Convert to JPEG
    success, encoded_img = cv2.imencode('.jpg', img)
    if not success:
        print("❌ Could not encode image")
        return

    # Send to server
    url = "http://localhost:8082/inspect"
    files = {'file': ('test_pcb.jpg', encoded_img.tobytes(), 'image/jpeg')}

    try:
        print("📤 Sending image to server...")
        response = requests.post(url, files=files, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"   Pass: {result['pass']}")
            print(f"   Defects found: {len(result['defects'])}")
            print(f"   Anomaly score: {result['anomaly_score']}")
            print(f"   Latency: {result['latency_ms']}ms")

            if result['defects']:
                print("   Detected defects:")
                for defect in result['defects']:
                    print(f"     - {defect['class']}: {defect['confidence']:.2f}")
            else:
                print("   No defects detected")

        else:
            print(f"❌ Server error: {response.status_code}")
            print(f"   Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_capture_analyze()