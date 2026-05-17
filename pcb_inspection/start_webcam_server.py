"""
Start PCB Inspection Server with Webcam Interface
=================================================
Runs the FastAPI server with webcam inspection capabilities.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Start the server."""
    print("🔍 PCB Inspection Server with Webcam Interface")
    print("=" * 55)

    # Check if models exist
    model_paths = [
        "./models/phase1/simple_anomaly_detector.pkl",
        "./models/phase2/yolov8_pcb_best.pt"
    ]

    for path in model_paths:
        if Path(path).exists():
            print(f"✅ Found model: {path}")
        else:
            print(f"⚠️  Missing model: {path}")

    print("\n🚀 Starting server...")
    print("📱 Webcam interface: http://localhost:8000/webcam")
    print("🔧 API docs: http://localhost:8000/docs")
    print("💡 Press Ctrl+C to stop\n")

    # Import and run server
    from server import app
    import uvicorn

    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()