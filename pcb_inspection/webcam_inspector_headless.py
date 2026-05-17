"""
PCB Webcam Inspector - Headless Version
========================================
Captures webcam frames and analyzes them without GUI display.
Saves results to files for inspection.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simplified_inspector import EnsembleInspector, SimplePCBInspector
import pickle


class HeadlessWebcamInspector:
    """Headless PCB inspection using webcam feed."""

    def __init__(self, model_path="./models/phase2/yolov8_pcb_best.pt"):
        """Initialize the headless webcam inspector."""
        self.model_path = Path(model_path)

        # Initialize webcam
        self.cap = None
        self.is_running = False

        # Load PCB inspection model
        self.inspector = None
        self.load_model()

        # Results storage
        self.results_dir = Path("./webcam_results")
        self.results_dir.mkdir(exist_ok=True)

        print("🎥 Headless Webcam PCB Inspector initialized!")
        print("📸 Press 'q' to quit, 's' to save current frame")
        print("🔍 Model loaded:", "YES" if self.inspector else "NO")

    def load_model(self):
        """Load the PCB inspection model."""
        try:
            # Load Phase 1 model (anomaly detection)
            pc_inspector = None
            pc_path = Path("./models/phase1/simple_anomaly_detector.pkl")
            if pc_path.exists():
                with open(pc_path, 'rb') as f:
                    model_data = pickle.load(f)
                pc_inspector = SimplePCBInspector(threshold=model_data['threshold'])
                pc_inspector.reference_stats = model_data['reference_stats']
                print("✅ Phase 1 model loaded")

            # Load Phase 2 model (YOLO defect detection)
            yolo_path = str(self.model_path) if self.model_path.exists() else None
            if yolo_path:
                print("✅ Phase 2 model loaded")
            else:
                print("⚠️  Phase 2 model not found - using anomaly detection only")

            self.inspector = EnsembleInspector(
                patchcore_path=None,
                yolo_path=yolo_path,
                pc_threshold=0.1,  # Lower threshold for webcam
                patchcore_inspector=pc_inspector
            )

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            self.inspector = None

    def start_webcam(self):
        """Start webcam capture."""
        self.cap = cv2.VideoCapture(0)  # Default webcam

        if not self.cap.isOpened():
            print("❌ Cannot access webcam!")
            print("💡 Make sure your webcam is not used by other applications")
            return False

        # Set webcam properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("📹 Webcam started successfully!")
        return True

    def draw_results_on_frame(self, frame, results):
        """Draw inspection results on the frame."""
        if not results or not results.get('defects'):
            return frame

        # Colors for different defect types
        colors = {
            'missing_component': (0, 0, 255),    # Red
            'solder_bridge': (255, 0, 0),        # Blue
            'cold_joint': (0, 255, 255),         # Yellow
            'trace_crack': (255, 0, 255),        # Magenta
            'contamination': (0, 165, 255),     # Orange
            'anomaly': (128, 128, 128)           # Gray
        }

        # Draw each detected defect
        for defect in results['defects']:
            # Get defect info
            defect_class = defect.get('class', 'unknown')
            confidence = defect.get('confidence', 0.0)
            bbox = defect.get('bbox', [0, 0, 100, 100])

            # Get color for this defect type
            color = colors.get(defect_class, (255, 255, 255))

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{defect_class}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), color, -1)

            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5),
                       font, font_scale, (255, 255, 255),
                       font_thickness)

        return frame

    def save_results(self, frame, results, frame_count):
        """Save current frame and results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"pcb_frame_{frame_count:04d}_{timestamp}"

        # Save original frame
        original_path = self.results_dir / f"{base_filename}_original.jpg"
        cv2.imwrite(str(original_path), frame)

        # Save annotated frame
        if results and results.get('defects'):
            annotated_frame = self.draw_results_on_frame(frame.copy(), results)
            annotated_path = self.results_dir / f"{base_filename}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_frame)

        # Save results as JSON
        json_path = self.results_dir / f"{base_filename}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved: {base_filename}")

    def print_status(self, frame_count, fps, results):
        """Print current status to console."""
        print(f"\rFrame: {frame_count:4d} | FPS: {fps:4.1f} | "
              f"Defects: {len(results.get('defects', [])) if results else 0:2d} | "
              f"Pass: {results.get('pass', True) if results else 'N/A'}", end='', flush=True)

    def run(self, duration_seconds=30):
        """Main webcam loop."""
        if not self.start_webcam():
            return

        self.is_running = True
        frame_count = 0
        start_time = time.time()
        end_time = start_time + duration_seconds

        print(f"\n🎬 Starting {duration_seconds}s webcam inspection!")
        print("📱 Position PCB in front of webcam")
        print("🎯 Press 'q' to quit, 's' to save current frame\n")

        try:
            while self.is_running and time.time() < end_time:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from webcam")
                    break

                frame_count += 1
                current_time = time.time()

                # Process frame every 3 frames (for smoother processing)
                results = None
                if frame_count % 3 == 0 and self.inspector:
                    try:
                        # Convert BGR to RGB for processing
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.inspector.inspect(rgb_frame)
                    except Exception as e:
                        print(f"⚠️  Processing error: {e}")

                # Calculate FPS
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Print status
                self.print_status(frame_count, fps, results)

                # Auto-save frames with defects
                if results and results.get('defects'):
                    self.save_results(frame, results, frame_count)

                # Check for quit command (simulate keyboard input)
                time.sleep(0.1)  # Small delay to prevent busy loop

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()

        print(f"\n✅ Inspection complete! Processed {frame_count} frames.")
        print(f"📁 Results saved to: {self.results_dir}")

    def cleanup(self):
        """Clean up resources."""
        self.is_running = False

        if self.cap:
            self.cap.release()

        print("🧹 Cleanup completed")


def main():
    """Main function."""
    print("\n🔍 PCB Webcam Inspector (Headless)")
    print("=" * 50)

    # Check if models exist
    model_path = "./models/phase2/yolov8_pcb_best.pt"
    if not Path(model_path).exists():
        print("⚠️  Warning: Trained YOLO model not found!")
        print("   Path:", model_path)
        print("   The app will still work with anomaly detection only.\n")

    # Create inspector
    inspector = HeadlessWebcamInspector(model_path)

    # Run webcam inspection for 30 seconds
    try:
        inspector.run(duration_seconds=30)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()