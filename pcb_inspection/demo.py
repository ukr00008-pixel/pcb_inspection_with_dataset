"""
PCB Inspection Demo — Local Streamlit App
==========================================
Simple local demo for PCB defect detection without Docker/Redis.

Usage:
    streamlit run demo.py
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
from PIL import Image

# Import our inspectors
from src.simplified_inspector import SimplePCBInspector, YOLOInspector, EnsembleInspector


def load_models():
    """Load trained models if available."""
    models = {}

    # Phase 1: Simple Anomaly Detection
    pc_path = Path("./models/phase1/simple_anomaly_detector.pkl")
    if pc_path.exists():
        try:
            import pickle

            # Load trained model
            with open(pc_path, 'rb') as f:
                model_data = pickle.load(f)

            inspector = SimplePCBInspector(threshold=model_data['threshold'])
            inspector.reference_stats = model_data['reference_stats']

            models['patchcore'] = inspector
            st.success("✅ Phase 1 (Simple Anomaly Detection) model loaded")
            st.info(f"Trained on {model_data['training_images']} good images")
        except Exception as e:
            st.error(f"❌ Failed to load Phase 1 model: {e}")
    else:
        st.warning("⚠️ Phase 1 model not found. Train it first with: python src/train_phase1_simple.py")

    # Phase 2: YOLOv8
    yolo_path = Path("./models/phase2/yolov8m.onnx")
    if yolo_path.exists():
        models['yolo'] = YOLOInspector(str(yolo_path))
        st.success("✅ Phase 2 (YOLOv8) model loaded")
    else:
        st.warning("⚠️ Phase 2 model not found. Train it first with: python src/train_phase2.py")

    return models


def main():
    st.set_page_config(page_title="PCB Inspection Demo", page_icon="🔍", layout="wide")

    st.title("🔍 PCB Defect Detection Demo")
    st.markdown("Upload a PCB image to detect defects using AI")

    # Load models
    models = load_models()

    # File uploader
    uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("🔍 Analysis Results")

            if not models:
                st.error("❌ No models available. Please train models first.")
                return

            # Create ensemble inspector
            inspector = EnsembleInspector(
                patchcore_path="./models/phase1/simple_anomaly_detector.pkl" if 'patchcore' in models else None,
                yolo_path="./models/phase2/yolov8m.onnx" if 'yolo' in models else None,
                pc_threshold=0.52
            )

            # Run inspection
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                result = inspector.inspect(img_bgr)
                latency = time.time() - start_time

            # Display results
            if result["pass"]:
                st.success("✅ PASS - No defects detected")
            else:
                st.error("❌ FAIL - Defects detected")

            st.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")
            st.metric("Processing Time", f"{latency:.2f}s")

            # Show detected defects
            if result["defects"]:
                st.subheader("🚨 Detected Defects")
                for defect in result["defects"]:
                    st.write(f"• **{defect['class']}** (confidence: {defect['confidence']:.2f})")

            # Show heatmap if available
            if "heatmap" in result and result["heatmap"] is not None:
                st.subheader("🔥 Anomaly Heatmap")
                # Create colored heatmap
                if 'patchcore' in models:
                    heatmap_img = models['patchcore'].get_colored_heatmap(
                        result["heatmap"], img_bgr)
                    heatmap_rgb = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
                    st.image(heatmap_rgb, use_column_width=True,
                            caption="Red areas indicate potential defects")

    # Sidebar with information
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown("""
    **PCB Defect Detection System**

    This demo uses two AI models:
    - **Phase 1**: PatchCore (unsupervised anomaly detection)
    - **Phase 2**: YOLOv8 (supervised defect classification)

    **Training Requirements:**
    - Phase 1: 200-500 good PCB images
    - Phase 2: 200+ labeled defect images per class
    """)

    st.sidebar.title("🚀 Quick Start")
    st.sidebar.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train Phase 1 (if you have good images)
python src/train_phase1.py

# 3. Run this demo
streamlit run demo.py
    """, language="bash")


if __name__ == "__main__":
    main()