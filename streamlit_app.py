"""
Streamlit PCB Inspection App
============================
Streamlit frontend for the PCB webcam inspection server.

Run:
  1) Start the backend server with `python simple_webcam_server.py`
  2) Run this app with `streamlit run streamlit_app.py`
"""

import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.simplified_inspector import EnsembleInspector

st.set_page_config(page_title="PCB Defect Inspector", page_icon="🔍", layout="wide")

st.title("🔍 PCB Defect Inspector (Streamlit)")
st.write(
    "Upload a PCB image or use the webcam to detect PCB defects using the trained model server."
)

pc_path = Path("./models/phase1/simple_anomaly_detector.pkl")
yolo_pt = Path("./models/phase2/yolov8_pcb_best.pt")
yolo_onnx = Path("./models/phase2/yolov8m.onnx")
remote_server_default = "http://localhost:8082/inspect"

mode = st.sidebar.radio(
    "Inspection Mode",
    ["Remote server (/inspect)", "Local direct inference"],
    index=0,
)

server_url = st.sidebar.text_input("Server URL", value=remote_server_default)

st.sidebar.header("Model Status")
if pc_path.exists():
    st.sidebar.success(f"Phase1: PatchCore found ({pc_path.name})")
else:
    st.sidebar.warning("Phase1: PatchCore not found")

if yolo_onnx.exists():
    st.sidebar.success(f"Phase2: YOLO ONNX found ({yolo_onnx.name})")
elif yolo_pt.exists():
    st.sidebar.info(f"Phase2: YOLO PyTorch found ({yolo_pt.name})")
else:
    st.sidebar.warning("Phase2: YOLO model not found")

if mode == "Remote server (/inspect)":
    st.sidebar.markdown(
        "Run `python simple_webcam_server.py` or `python app.py`, then enter the server URL above."
    )
else:
    st.sidebar.markdown("Local direct inference will use the models loaded inside Streamlit.")

show_graph = st.sidebar.checkbox("Show model pipeline graph", value=True)


def build_model_dot():
    edges = []

    if pc_path.exists():
        edges.append(('"Input Image"', '"PatchCore (Phase 1)"'))

    if yolo_onnx.exists():
        yolo_node = '"YOLO (ONNX - Phase 2)"'
        edges.append(('"Input Image"', yolo_node))
    elif yolo_pt.exists():
        yolo_node = '"YOLO (PyTorch - Phase 2)"'
        edges.append(('"Input Image"', yolo_node))
    else:
        yolo_node = None

    if pc_path.exists():
        edges.append(('"PatchCore (Phase 1)"', '"Ensemble Inspector"'))
    if yolo_node:
        edges.append((yolo_node, '"Ensemble Inspector"'))
    edges.append(('"Ensemble Inspector"', '"Decision (Pass/Fail)"'))

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        "  node [shape=box, style=filled, fillcolor=\"#eef3f8\"];",
    ]
    for a, b in edges:
        lines.append(f"  {a} -> {b};")
    lines.append("}")
    return "\n".join(lines)


if show_graph:
    st.subheader("Model Pipeline")
    try:
        st.graphviz_chart(build_model_dot())
    except Exception as e:
        st.error(f"Failed to render model graph: {e}")


def load_inspector():
    patchcore = str(pc_path) if pc_path.exists() else None
    yolo_path = str(yolo_onnx) if yolo_onnx.exists() else (str(yolo_pt) if yolo_pt.exists() else None)

    return EnsembleInspector(
        patchcore_path=patchcore,
        yolo_path=yolo_path,
        pc_threshold=0.52,
    )


def post_image_to_server(url: str, image_bytes: bytes, filename: str = "pcb.jpg") -> dict:
    boundary = f"----StreamlitBoundary{int(time.time() * 1000)}"
    body = BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(image_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(url, data=body.getvalue(), method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    request.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_text = response.read().decode("utf-8")
            return json.loads(response_text)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Server error {exc.code}: {exc.read().decode('utf-8')}" ) from exc


def annotate_image(img_bgr: np.ndarray, defects: list) -> np.ndarray:
    output = img_bgr.copy()
    for defect in defects or []:
        bbox = defect.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            label = defect.get("class", "defect")
            conf = defect.get("confidence", 0.0)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                output,
                f"{label} {conf:.2f}",
                (x1, max(10, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
    return output


def image_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = pil_image.convert("RGB")
    array = np.array(rgb)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


input_mode = st.radio("Input source", ["Upload image", "Live webcam"], index=1)

uploaded_file = None
camera_input = None

if input_mode == "Upload image":
    uploaded_file = st.file_uploader("Upload a PCB image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image to start inspection.")

if input_mode == "Live webcam":
    st.markdown("### Live webcam")
    camera_input = st.camera_input("Use your webcam to capture a PCB frame")
    if camera_input is None:
        st.info("Allow webcam access and capture a frame to analyze.")

analyze_button = st.button("Analyze")
inspector = None

image_bytes = None
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
elif camera_input is not None:
    image_bytes = camera_input.read()

if image_bytes is None and analyze_button:
    st.warning("No image available for analysis. Capture from the webcam or upload a file.")

if image_bytes is not None and analyze_button:
    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_bgr = image_to_bgr(pil_image)
        bytes_buf = BytesIO()
        pil_image.save(bytes_buf, format="JPEG")
        image_bytes = bytes_buf.getvalue()

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Original")
            st.image(pil_image, width=380)

        with col2:
            st.subheader("Analysis")
            if mode == "Remote server (/inspect)":
                try:
                    start_ts = time.time()
                    result = post_image_to_server(server_url, image_bytes)
                    result["latency_ms"] = int((time.time() - start_ts) * 1000)
                except Exception as exc:
                    st.error(f"Remote inspection failed: {exc}")
                    result = None
            else:
                if inspector is None:
                    with st.spinner("Loading local models..."):
                        inspector = load_inspector()
                try:
                    start_ts = time.time()
                    result = inspector.inspect(img_bgr)
                    result["latency_ms"] = int((time.time() - start_ts) * 1000)
                except Exception as exc:
                    st.error(f"Local inspection failed: {exc}")
                    result = None

            if result is not None:
                if result.get("pass"):
                    st.success("✅ PASS - No defects detected")
                else:
                    st.error("❌ FAIL - Defects detected")

                st.write(f"Anomaly score: {result.get('anomaly_score', 0):.4f}")
                st.write(f"Latency: {result.get('latency_ms', 0)} ms")

                annotated = annotate_image(img_bgr, result.get("defects", []))
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Annotated output", width=380)

                defects = result.get("defects") or []
                if defects:
                    st.subheader("Detected Defects")
                    for defect in defects:
                        st.write(
                            f"- **{defect.get('class','unknown')}** — confidence: {defect.get('confidence', 0):.2f}"
                        )
            else:
                st.warning("No result available. Check model/server logs and verify the server URL.")

    except Exception as exc:
        st.error(f"Could not process image: {exc}")

st.markdown("---")
st.caption("Tip: Place the PCB centrally and use clear lighting for best results.")
