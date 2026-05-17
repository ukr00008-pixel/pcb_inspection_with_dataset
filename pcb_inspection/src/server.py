"""
PCB Inspection API — FastAPI Production Server
===============================================
Endpoints:
  POST /inspect  — Upload PCB image, returns PASS/FAIL + defect list
  GET  /health   — Liveness check
  GET  /metrics  — Aggregated stats from Redis stream

Latency target: < 200ms P99

Usage:
    python src/server.py
    # or via Docker Compose
"""

import cv2
import json
import numpy as np
import redis
import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Paths can be overridden via environment variables
PATCHCORE_PATH = os.getenv("PATCHCORE_PATH", "./models/phase1/simple_anomaly_detector.pkl")
YOLO_PATH      = os.getenv("YOLO_PATH",      "./models/phase2/yolov8_pcb_best.pt")
PC_THRESHOLD   = float(os.getenv("PC_THRESHOLD", "0.1"))
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))


# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and connect to Redis on startup."""
    from simplified_inspector import EnsembleInspector

    print("[Startup] Loading models...")
    app.state.inspector = EnsembleInspector(
        patchcore_path=PATCHCORE_PATH,
        yolo_path=YOLO_PATH,
        pc_threshold=PC_THRESHOLD,
    )

    print(f"[Startup] Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    app.state.redis = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    app.state.redis.ping()

    print("[Startup] ✓ Ready for inspection")
    yield

    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="PCB Inspection API",
    version="2.0",
    description="Real-time PCB defect detection via PatchCore + YOLOv8 ensemble",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check — returns 200 if server is up and models are loaded."""
    return {"status": "ok", "version": "2.0"}


@app.get("/webcam")
async def webcam_interface():
    """Serve the webcam inspection interface."""
    try:
        with open("webcam_interface.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Webcam interface not found")


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

    # Run ensemble inspection
    result = app.state.inspector.inspect(img)

    latency_ms = (time.perf_counter() - t0) * 1000

    # Log to Redis Stream (maxlen=10000 keeps last 10k results)
    try:
        app.state.redis.xadd("inspections", {
            "pass":     str(result["pass"]),
            "score":    str(result["anomaly_score"]),
            "defects":  json.dumps(result["defects"]),
            "latency":  str(latency_ms),
        }, maxlen=10000)
    except redis.RedisError as e:
        print(f"[Warning] Redis log failed: {e}")

    # Strip heatmap array from response (not JSON-serializable)
    return {
        "pass":          result["pass"],
        "defects":       result["defects"],
        "anomaly_score": round(result["anomaly_score"], 4),
        "latency_ms":    round(latency_ms, 1),
    }


@app.get("/metrics")
async def metrics():
    """
    Aggregated inspection statistics from the last 1000 results.

    Returns total count, defect rate, and latency percentiles.
    """
    try:
        entries = app.state.redis.xrevrange("inspections", count=1000)
    except redis.RedisError:
        return {"error": "Redis unavailable"}

    total = len(entries)
    if not total:
        return {"total_inspected": 0, "message": "No inspections logged yet"}

    failures = sum(1 for _, e in entries if e[b"pass"] == b"False")
    lats = [float(e[b"latency"]) for _, e in entries]

    return {
        "total_inspected": total,
        "defect_rate_pct": round(failures / total * 100, 2),
        "pass_rate_pct":   round((total - failures) / total * 100, 2),
        "avg_latency_ms":  round(np.mean(lats), 1),
        "p50_latency_ms":  round(np.percentile(lats, 50), 1),
        "p95_latency_ms":  round(np.percentile(lats, 95), 1),
        "p99_latency_ms":  round(np.percentile(lats, 99), 1),
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,        # Single worker — GPU models are not fork-safe
        reload=False,
    )
