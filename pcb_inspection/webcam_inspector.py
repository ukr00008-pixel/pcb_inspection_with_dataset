"""
Real-time PCB Inspection with Webcam
=====================================
Live PCB defect detection using your laptop webcam.
Shows defects with bounding boxes and labels in real-time.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class WebcamPCBInspector:
    """Real-time PCB inspection using webcam feed."""

    def __init__(self, model_path="./models/phase2/yolov8_pcb_best.pt"):
        """Initialize the webcam inspector."""
        self.model_path = Path(model_path)

        # Initialize webcam
        self.cap = None
        self.is_running = False

        # Load PCB inspection model
        self.inspector = None
        self.load_model()

        # Display settings
        self.window_name = "PCB Defect Inspector - Webcam"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Colors for different defect types
        self.colors = {
            'missing_component': (0, 0, 255),    # Red
            'solder_bridge': (255, 0, 0),        # Blue
            'cold_joint': (0, 255, 255),         # Yellow
            'trace_crack': (255, 0, 255),        # Magenta
            'contamination': (0, 165, 255),     # Orange
            'anomaly': (128, 128, 128)           # Gray
        }

        print("🎥 Webcam PCB Inspector initialized!")
        print("📸 Press 'q' to quit, 's' to save current frame")
        print("🔍 Model loaded:", "YES" if self.inspector else "NO")

    def load_model(self):
        """Load the PCB inspection model."""
        try:
            from simplified_inspector import EnsembleInspector, SimplePCBInspector
            import pickle

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

    def draw_results(self, frame, results):
        """Draw inspection results on the frame."""
        if not results or not results.get('defects'):
            return frame

        # Draw each detected defect
        for defect in results['defects']:
            # Get defect info
            defect_class = defect.get('class', 'unknown')
            confidence = defect.get('confidence', 0.0)
            bbox = defect.get('bbox', [0, 0, 100, 100])

            # Get color for this defect type
            color = self.colors.get(defect_class, (255, 255, 255))

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{defect_class}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10),
                         (x1 + text_width, y1), color, -1)

            # Draw text
            cv2.putText(frame, label, (x1, y1 - 5),
                       self.font, self.font_scale, (255, 255, 255),
                       self.font_thickness)

        return frame

    def add_status_overlay(self, frame, fps, results):
        """Add status information overlay."""
        height, width = frame.shape[:2]

        # Status text
        status_lines = [
            f"FPS: {fps:.1f}",
            f"Defects: {len(results.get('defects', [])) if results else 0}",
            f"Pass: {results.get('pass', True) if results else 'N/A'}",
            f"Model: {'Loaded' if self.inspector else 'Missing'}"
        ]

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw status text
        for i, line in enumerate(status_lines):
            cv2.putText(frame, line, (20, 35 + i * 25),
                       self.font, 0.6, (255, 255, 255), 1)

        # Instructions
        instructions = [
            "Press 'q' to quit",
            "Press 's' to save frame",
            "Press 'c' to capture PCB"
        ]

        for i, instr in enumerate(instructions):
            cv2.putText(frame, instr, (width - 250, 35 + i * 25),
                       self.font, 0.5, (200, 200, 200), 1)

        return frame

    def save_frame(self, frame, results):
        """Save current frame with results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pcb_capture_{timestamp}.jpg"

        # Add results info to saved image
        save_frame = frame.copy()
        if results and results.get('defects'):
            save_frame = self.draw_results(save_frame, results)

        cv2.imwrite(filename, save_frame)
        print(f"💾 Frame saved: {filename}")

    def run(self):
        """Main webcam loop."""
        if not self.start_webcam():
            return

        self.is_running = True
        frame_count = 0
        start_time = time.time()
        last_process_time = 0

        print("\n🎬 Starting real-time PCB inspection!")
        print("📱 Position PCB in front of webcam")
        print("🎯 Press 'q' to quit, 's' to save current frame\n")

        try:
            while self.is_running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from webcam")
                    break

                frame_count += 1
                current_time = time.time()

                # Process frame every 3 frames (for smoother display)
                results = None
                if frame_count % 3 == 0 and self.inspector:
                    try:
                        # Convert BGR to RGB for processing
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.inspector.inspect(rgb_frame)
                        last_process_time = current_time
                    except Exception as e:
                        print(f"⚠️  Processing error: {e}")

                # Calculate FPS
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Draw results on frame
                if results:
                    frame = self.draw_results(frame, results)

                # Add status overlay
                frame = self.add_status_overlay(frame, fps, results)

                # Show frame
                cv2.imshow(self.window_name, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):  # Quit
                    print("👋 Quitting...")
                    break
                elif key == ord('s'):  # Save frame
                    self.save_frame(frame, results)
                elif key == ord('c'):  # Capture PCB
                    print("📸 PCB capture mode - analyzing...")
                    # Could add special capture logic here

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.is_running = False

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        print("🧹 Cleanup completed")


def main():
    """Main function."""
    print("\n🔍 PCB Webcam Inspector")
    print("=" * 50)

    # Check if models exist
    model_path = "./models/phase2/yolov8_pcb_best.pt"
    if not Path(model_path).exists():
        print("⚠️  Warning: Trained YOLO model not found!")
        print("   Path:", model_path)
        print("   The app will still work with anomaly detection only.\n")

    # Create inspector
    inspector = WebcamPCBInspector(model_path)

    # Run webcam inspection
    try:
        inspector.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
