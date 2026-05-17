"""
PCB Inspection API — Local FastAPI Server (No Docker/Redis)
===========================================================
Simple local server for testing the PCB inspection API.

Usage:
    python src/local_server.py
    # API available at http://localhost:8000
    # Test with: curl -X POST http://localhost:8000/inspect -F "file=@image.jpg"
"""

import cv2
import numpy as np
import time
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Paths - can be overridden via environment variables
PATCHCORE_PATH = os.getenv("PATCHCORE_PATH", "./models/phase1/simple_anomaly_detector.pkl")
YOLO_PATH      = os.getenv("YOLO_PATH",      "./models/phase2/yolov8_pcb_best.pt")
PC_THRESHOLD   = float(os.getenv("PC_THRESHOLD", "0.1"))


def create_inspector():
    """Create ensemble inspector, handling missing models gracefully."""
    from simplified_inspector import EnsembleInspector, SimplePCBInspector
    import pickle

    pc_path = Path(PATCHCORE_PATH)
    yolo_path = Path(YOLO_PATH)

    # For simplified inspector, we need to load the trained model
    patchcore_inspector = None
    if pc_path.exists():
        try:
            with open(pc_path, 'rb') as f:
                model_data = pickle.load(f)

            patchcore_inspector = SimplePCBInspector(threshold=model_data['threshold'])
            patchcore_inspector.reference_stats = model_data['reference_stats']
            print(f"[Server] Loaded Phase 1 model: {pc_path}")
        except Exception as e:
            print(f"[Server] Failed to load Phase 1 model: {e}")

    yolo_path_str = str(yolo_path) if yolo_path.exists() else None

    return EnsembleInspector(
        patchcore_path=None,  # We pass the inspector object instead
        yolo_path=yolo_path_str,
        pc_threshold=PC_THRESHOLD,
        patchcore_inspector=patchcore_inspector
    )


# Global inspector (lazy-loaded)
inspector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    global inspector
    try:
        print("[Startup] Loading models...")
        inspector = create_inspector()
        print("[Startup] ✓ Ready for inspection")
    except Exception as e:
        print(f"[Startup] ❌ Model loading failed: {e}")
        print("[Startup] Server will start but inspections will fail")

    yield

    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="PCB Inspection API (Local)",
    version="1.0",
    description="Local PCB defect detection API - no Docker/Redis required",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Liveness check."""
    if inspector is None:
        return {"status": "error", "message": "Models not loaded"}
    return {"status": "ok", "version": "1.0"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PCB Inspection API (Local)",
        "version": "1.0",
        "description": "Local PCB defect detection API - no Docker/Redis required",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "POST /inspect": "Inspect PCB image for defects",
            "GET /docs": "Interactive API documentation"
        },
        "usage": "Upload a PCB image via POST /inspect to detect defects"
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    """
    Inspect a PCB image for defects.

    Upload a JPEG/PNG image of a PCB board.

    Returns:
        pass        : bool — True = no defects detected
        defects     : list of detected defects with class, confidence, bbox
        anomaly_score : float (0-1) — PatchCore score
        latency_ms  : float — end-to-end inference time
    """
    if inspector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t0 = time.perf_counter()

    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Use JPEG or PNG.")

    # Decode image
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Run inspection
    result = inspector.inspect(img)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Return result (no Redis logging)
    return {
        "pass":          result["pass"],
        "defects":       result["defects"],
        "anomaly_score": round(result["anomaly_score"], 4),
        "latency_ms":    round(latency_ms, 1),
    }


if __name__ == "__main__":
    print("🚀 Starting PCB Inspection API (Local)")
    print("📡 API will be available at: http://localhost:8001")
    print("🛑 Press Ctrl+C to stop")

    uvicorn.run(
        "local_server:app",
        host="localhost",
        port=8001,  # Changed from 8000 to 8001
        reload=False,
        log_level="info"
    )