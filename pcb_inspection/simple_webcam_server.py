"""
Simple PCB Webcam Server
========================
Basic HTTP server to serve the webcam interface.
"""

import http.server
import socketserver
import webbrowser
import threading
import time
from pathlib import Path
import os
import json
import cv2
import numpy as np
import sys
from io import BytesIO

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from simplified_inspector import EnsembleInspector, SimplePCBInspector
import pickle

PORT = 8082

# Global inspector instance
inspector = None

class PCBHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for PCB inspection requests."""

    def __init__(self, *args, **kwargs):
        global inspector
        if inspector is None:
            inspector = self.load_inspector()
        super().__init__(*args, **kwargs)

    def load_inspector(self):
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
            yolo_path = "./models/phase2/yolov8_pcb_best.pt"
            if Path(yolo_path).exists():
                print("✅ Phase 2 model loaded")
            else:
                yolo_path = None
                print("⚠️  Phase 2 model not found")

            inspector = EnsembleInspector(
                patchcore_path=None,
                yolo_path=yolo_path,
                pc_threshold=0.1,
                patchcore_inspector=pc_inspector
            )
            return inspector

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/':
            self.path = '/webcam_interface.html'
        elif self.path == '/webcam':
            self.path = '/webcam_interface.html'

        return super().do_GET()

    def do_POST(self):
        """Handle POST requests for inspection."""
        if self.path == '/inspect':
            self.handle_inspect()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_inspect(self):
        """Handle PCB inspection requests."""
        print("📥 Received inspection request")
        try:
            # Get content length
            content_length = int(self.headers['Content-Length'])
            print(f"📏 Content length: {content_length}")

            # Read the raw POST data
            post_data = self.rfile.read(content_length)
            print(f"📦 Received {len(post_data)} bytes of data")

            # Simple multipart parser - find the image data
            boundary = None
            content_type = self.headers.get('Content-Type', '')
            print(f"📄 Content type: {content_type}")
            if 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1].strip()

            if not boundary:
                print("❌ No boundary found")
                self.send_error(400, "Invalid multipart data")
                return

            print(f"🔀 Boundary: {boundary}")
            # Split by boundary
            boundary_bytes = b'--' + boundary.encode()
            parts = post_data.split(boundary_bytes)
            print(f"📊 Found {len(parts)} parts")

            image_data = None
            for i, part in enumerate(parts):
                if b'Content-Type: image/jpeg' in part or b'Content-Type: image/png' in part:
                    print(f"🖼️ Found image in part {i}")
                    # Find the actual image data (after headers)
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        image_data = part[header_end + 4:-2]  # Remove \r\n at end
                        print(f"📷 Extracted image data: {len(image_data)} bytes")
                        break

            if image_data is None:
                print("❌ No image data found")
                self.send_error(400, "No image found in request")
                return

            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("❌ Could not decode image")
                self.send_error(400, "Could not decode image")
                return

            print(f"🖼️ Decoded image shape: {img.shape}")
            # Run inspection
            if inspector is None:
                print("❌ Inspector not loaded")
                self.send_error(500, "Inspector not loaded")
                return

            print("🔍 Processing image of size: {img.shape}")
            # Convert BGR to RGB for processing
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("🔄 Running inspection...")
            result = inspector.inspect(rgb_img)
            print(f"✅ Inspection complete. Pass: {result['pass']}, Defects: {len(result['defects'])}")

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {
                "pass": result["pass"],
                "defects": result["defects"],
                "anomaly_score": round(float(result["anomaly_score"]), 4),
                "latency_ms": 150  # Mock latency
            }

            self.wfile.write(json.dumps(response).encode())
            print("📤 Response sent")

        except Exception as e:
            print(f"❌ Inspection error: {e}")
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Inspection failed: {str(e)}")

    def end_headers(self):
        """Add CORS headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """Start the simple HTTP server."""
    # Change to the current directory
    web_dir = Path(__file__).parent
    os.chdir(web_dir)

    with socketserver.TCPServer(("", PORT), PCBHandler) as httpd:
        print("🌐 Server started at http://localhost:{PORT}")
        print("📱 Webcam interface: http://localhost:8082/webcam")
        print("🔧 Opening browser automatically...")
        print("💡 Press Ctrl+C to stop\n")

        # Open browser after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f"http://localhost:{PORT}/webcam")

        threading.Thread(target=open_browser, daemon=True).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    print("🔍 PCB Webcam Server")
    print("=" * 30)
    start_server()