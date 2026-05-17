"""
Synthetic PCB Dataset Generator
================================
Creates realistic-looking PCB images (good + 5 defect classes) using
OpenCV drawing primitives — no real hardware needed.

Generates:
  data/pcb/good/              — 300 clean PCB images (Phase 1 training)
  data/pcb_labeled/images/    — 500 defect images across train/val/test
  data/pcb_labeled/labels/    — YOLO .txt annotation files

Run: python generate_dataset.py
"""

import cv2
import numpy as np
import random
import os
from pathlib import Path


# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

IMG_SIZE = 640

# ── Color palette (BGR) ───────────────────────────────────────────────────────
PCB_GREEN       = (34,  85,  34)    # FR4 substrate
COPPER          = (30, 120, 200)    # Copper traces (slightly oxidized)
SOLDER_GOOD     = (180, 200, 210)   # Shiny solder
SOLDER_COLD     = (120, 130, 130)   # Dull/grainy cold joint
SOLDER_BRIDGE   = (160, 190, 210)   # Bridge blob
IC_BODY         = (30,  30,  30)    # IC package
IC_PIN          = (40, 130, 190)    # IC pin/lead
RESIST_MASK     = (20,  60,  20)    # Solder-resist areas
PAD_COLOR       = (20, 140, 220)    # Exposed copper pads
CONTAMINATION_C = (50,  40, 110)    # Flux residue / contamination


# ═════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL DRAWING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def add_pcb_texture(img: np.ndarray) -> np.ndarray:
    """Add subtle fiber-glass weave texture to the PCB substrate."""
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def draw_substrate(img: np.ndarray) -> np.ndarray:
    """Fill background with PCB green + texture."""
    img[:] = PCB_GREEN
    return add_pcb_texture(img)


def draw_grid_traces(img: np.ndarray) -> np.ndarray:
    """Draw a grid of horizontal + vertical copper traces."""
    h, w = img.shape[:2]
    # Horizontal traces
    for y in range(60, h - 60, random.randint(55, 80)):
        thickness = random.choice([2, 3, 4])
        cv2.line(img, (30, y), (w - 30, y), COPPER, thickness)
    # Vertical traces
    for x in range(60, w - 60, random.randint(60, 90)):
        thickness = random.choice([2, 3, 4])
        cv2.line(img, (x, 30), (x, h - 30), COPPER, thickness)
    return img


def draw_ic_component(img: np.ndarray, cx: int, cy: int,
                      w: int = 80, h: int = 60) -> tuple:
    """Draw a rectangular IC package with pins on both sides. Returns bbox."""
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), IC_BODY, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), 1)
    # Pins left side
    pin_count = random.randint(3, 6)
    step = h // (pin_count + 1)
    for i in range(1, pin_count + 1):
        py = y1 + i * step
        cv2.rectangle(img, (x1 - 8, py - 3), (x1, py + 3), IC_PIN, -1)
        cv2.rectangle(img, (x2, py - 3), (x2 + 8, py + 3), IC_PIN, -1)
    return (x1 - 8, y1, x2 + 8, y2)


def draw_resistor(img: np.ndarray, cx: int, cy: int) -> tuple:
    """Draw a small SMD resistor (0402/0603 style)."""
    w, h = random.randint(20, 30), random.randint(10, 16)
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    # Body
    body_color = tuple(int(c) for c in np.random.randint(40, 100, 3).tolist())
    cv2.rectangle(img, (x1, y1), (x2, y2), body_color, -1)
    # End caps (pads)
    cv2.rectangle(img, (x1, y1), (x1 + 6, y2), PAD_COLOR, -1)
    cv2.rectangle(img, (x2 - 6, y1), (x2, y2), PAD_COLOR, -1)
    return (x1, y1, x2, y2)


def draw_capacitor(img: np.ndarray, cx: int, cy: int) -> tuple:
    """Draw a small SMD capacitor."""
    r = random.randint(8, 14)
    color = tuple(int(c) for c in [random.randint(80, 160),
                                    random.randint(80, 120),
                                    random.randint(20, 60)])
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(img, (cx, cy), r, (60, 60, 60), 1)
    return (cx - r, cy - r, cx + r, cy + r)


def draw_solder_joint(img: np.ndarray, cx: int, cy: int,
                      good: bool = True) -> None:
    """Draw a solder joint — shiny if good, dull/grainy if cold."""
    r = random.randint(5, 9)
    color = SOLDER_GOOD if good else SOLDER_COLD
    cv2.circle(img, (cx, cy), r, color, -1)
    if good:
        # Specular highlight
        cv2.circle(img, (cx - 2, cy - 2), r // 3, (230, 240, 245), -1)
    else:
        # Grainy texture
        for _ in range(8):
            gx = cx + random.randint(-r, r)
            gy = cy + random.randint(-r, r)
            cv2.circle(img, (gx, gy), 1, (100, 110, 110), -1)


def place_components(img: np.ndarray) -> list:
    """Place a realistic grid of ICs, resistors, capacitors. Returns component list."""
    components = []
    # ICs in a loose grid
    for row in range(2):
        for col in range(3):
            cx = 120 + col * 160 + random.randint(-15, 15)
            cy = 140 + row * 220 + random.randint(-15, 15)
            bbox = draw_ic_component(img, cx, cy,
                                     w=random.randint(70, 100),
                                     h=random.randint(50, 70))
            components.append(("ic", cx, cy, bbox))

    # Resistors scattered around
    for _ in range(12):
        cx = random.randint(60, IMG_SIZE - 60)
        cy = random.randint(60, IMG_SIZE - 60)
        bbox = draw_resistor(img, cx, cy)
        components.append(("resistor", cx, cy, bbox))

    # Capacitors
    for _ in range(8):
        cx = random.randint(60, IMG_SIZE - 60)
        cy = random.randint(60, IMG_SIZE - 60)
        bbox = draw_capacitor(img, cx, cy)
        components.append(("capacitor", cx, cy, bbox))

    return components


def draw_solder_joints(img: np.ndarray, components: list) -> None:
    """Draw good solder joints at each component pad."""
    for ctype, cx, cy, bbox in components:
        if ctype == "ic":
            x1, y1, x2, y2 = bbox
            # Left-side pads
            for py in range(y1 + 10, y2, 14):
                draw_solder_joint(img, x1 + 4, py, good=True)
            # Right-side pads
            for py in range(y1 + 10, y2, 14):
                draw_solder_joint(img, x2 - 4, py, good=True)
        elif ctype == "resistor":
            draw_solder_joint(img, bbox[0] + 4, cy, good=True)
            draw_solder_joint(img, bbox[2] - 4, cy, good=True)
        elif ctype == "capacitor":
            draw_solder_joint(img, cx - bbox[2] + cx - 2, cy, good=True)
            draw_solder_joint(img, cx + bbox[2] - cx + 2, cy, good=True)


# ═════════════════════════════════════════════════════════════════════════════
# GOOD BOARD GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def make_good_board(size: int = IMG_SIZE) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    draw_substrate(img)
    draw_grid_traces(img)
    components = place_components(img)
    draw_solder_joints(img, components)
    # Slight global blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# ═════════════════════════════════════════════════════════════════════════════
# DEFECT INJECTORS  — return (image, [yolo_annotations])
# ═════════════════════════════════════════════════════════════════════════════

def _yolo_box(cx, cy, w, h, cls_id, img_size=IMG_SIZE):
    """Convert pixel bbox to YOLO normalized format."""
    xc = np.clip(cx / img_size, 0, 1)
    yc = np.clip(cy / img_size, 0, 1)
    bw = np.clip(w / img_size, 0.01, 1)
    bh = np.clip(h / img_size, 0.01, 1)
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


# ── Class 0: missing_component ────────────────────────────────────────────────
def inject_missing_component(img: np.ndarray) -> tuple:
    annotations = []
    # Pick a random rectangular region and erase the component (fill with substrate)
    for _ in range(random.randint(1, 3)):
        cx = random.randint(100, IMG_SIZE - 100)
        cy = random.randint(100, IMG_SIZE - 100)
        w  = random.randint(40, 90)
        h  = random.randint(30, 70)
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        # Erase — show only substrate + pads (component is gone)
        cv2.rectangle(img, (x1, y1), (x2, y2), PCB_GREEN, -1)
        add_pcb_texture(img[y1:y2, x1:x2])
        # Empty pads still visible
        for px in [x1 + 8, x2 - 8]:
            cv2.rectangle(img, (px - 5, cy - 4), (px + 5, cy + 4), PAD_COLOR, -1)
        annotations.append(_yolo_box(cx, cy, w + 20, h + 20, cls_id=0))
    return img, annotations


# ── Class 1: solder_bridge ────────────────────────────────────────────────────
def inject_solder_bridge(img: np.ndarray) -> tuple:
    annotations = []
    for _ in range(random.randint(1, 3)):
        # Two pads close together bridged by solder blob
        cx = random.randint(80, IMG_SIZE - 80)
        cy = random.randint(80, IMG_SIZE - 80)
        gap = random.randint(8, 18)
        # Pad 1
        cv2.rectangle(img, (cx - 20, cy - 5), (cx - gap, cy + 5), PAD_COLOR, -1)
        # Pad 2
        cv2.rectangle(img, (cx + gap, cy - 5), (cx + 20, cy + 5), PAD_COLOR, -1)
        # Bridge blob
        bridge_w = random.randint(gap + 10, gap + 25)
        cv2.ellipse(img, (cx, cy), (bridge_w // 2, random.randint(5, 10)),
                    0, 0, 360, SOLDER_BRIDGE, -1)
        bw, bh = bridge_w + 15, 22
        annotations.append(_yolo_box(cx, cy, bw, bh, cls_id=1))
    return img, annotations


# ── Class 2: cold_joint ───────────────────────────────────────────────────────
def inject_cold_joint(img: np.ndarray) -> tuple:
    annotations = []
    for _ in range(random.randint(2, 5)):
        cx = random.randint(60, IMG_SIZE - 60)
        cy = random.randint(60, IMG_SIZE - 60)
        r  = random.randint(7, 12)
        # Dull, grainy solder
        cv2.circle(img, (cx, cy), r, SOLDER_COLD, -1)
        # Grain texture
        for _ in range(12):
            gx = cx + random.randint(-r + 2, r - 2)
            gy = cy + random.randint(-r + 2, r - 2)
            cv2.circle(img, (gx, gy), random.randint(1, 2), (90, 100, 100), -1)
        # No highlight (dull surface)
        size = r * 2 + 10
        annotations.append(_yolo_box(cx, cy, size, size, cls_id=2))
    return img, annotations


# ── Class 3: trace_crack ──────────────────────────────────────────────────────
def inject_trace_crack(img: np.ndarray) -> tuple:
    annotations = []
    for _ in range(random.randint(1, 3)):
        # Draw a trace, then interrupt it with substrate color
        x1 = random.randint(60, IMG_SIZE // 2)
        y  = random.randint(60, IMG_SIZE - 60)
        x2 = x1 + random.randint(60, 120)
        thickness = random.choice([3, 4])
        # Draw trace
        cv2.line(img, (x1, y), (x2, y), COPPER, thickness)
        # Crack — erase a segment
        crack_x = random.randint(x1 + 15, x2 - 15)
        crack_w = random.randint(4, 12)
        cv2.rectangle(img,
                      (crack_x - crack_w // 2, y - thickness - 2),
                      (crack_x + crack_w // 2, y + thickness + 2),
                      PCB_GREEN, -1)
        # Jagged edges
        for dx in range(-2, 3):
            cv2.line(img,
                     (crack_x + dx, y - thickness),
                     (crack_x + dx + random.randint(-2, 2), y + thickness),
                     (20, 50, 20), 1)
        bw, bh = crack_w + 30, 20
        annotations.append(_yolo_box(crack_x, y, bw, bh, cls_id=3))
    return img, annotations


# ── Class 4: contamination ────────────────────────────────────────────────────
def inject_contamination(img: np.ndarray) -> tuple:
    annotations = []
    for _ in range(random.randint(1, 4)):
        cx = random.randint(60, IMG_SIZE - 60)
        cy = random.randint(60, IMG_SIZE - 60)
        # Irregular flux residue blob
        pts = []
        r_base = random.randint(12, 30)
        for angle in range(0, 360, 20):
            r = r_base + random.randint(-6, 6)
            px = int(cx + r * np.cos(np.radians(angle)))
            py = int(cy + r * np.sin(np.radians(angle)))
            pts.append([px, py])
        pts = np.array(pts, dtype=np.int32)
        # Semi-transparent look
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], CONTAMINATION_C)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        bw = bh = r_base * 2 + 20
        annotations.append(_yolo_box(cx, cy, bw, bh, cls_id=4))
    return img, annotations


DEFECT_INJECTORS = {
    0: inject_missing_component,
    1: inject_solder_bridge,
    2: inject_cold_joint,
    3: inject_trace_crack,
    4: inject_contamination,
}

CLASS_NAMES = [
    "missing_component", "solder_bridge", "cold_joint",
    "trace_crack", "contamination"
]


# ═════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def make_defect_board(cls_id: int) -> tuple:
    """Create a defect board for a given class. Returns (image, annotations)."""
    img = make_good_board()
    img, annotations = DEFECT_INJECTORS[cls_id](img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, annotations


def generate_good_images(out_dir: str, count: int = 300):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Generating {count} good board images → {out_dir}")
    for i in range(count):
        img = make_good_board()
        cv2.imwrite(f"{out_dir}/good_{i:04d}.jpg", img)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{count} done")
    print(f"  ✓ {count} good images saved")


def generate_defect_images(images_dir: str, labels_dir: str,
                           count_per_class: int = 100):
    """
    Generate defect images split 70/20/10 across train/val/test.
    Each class gets `count_per_class` images.
    """
    splits = {
        "train": int(count_per_class * 0.70),
        "val":   int(count_per_class * 0.20),
        "test":  count_per_class - int(count_per_class * 0.70) - int(count_per_class * 0.20),
    }

    total = 0
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        print(f"  Class {cls_id}: {cls_name}")
        idx = 0
        for split, n in splits.items():
            img_out = Path(images_dir) / split
            lbl_out = Path(labels_dir) / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for _ in range(n):
                img, annotations = make_defect_board(cls_id)
                stem = f"{cls_name}_{idx:04d}"
                cv2.imwrite(str(img_out / f"{stem}.jpg"), img)
                (lbl_out / f"{stem}.txt").write_text("\n".join(annotations))
                idx += 1
                total += 1
        print(f"    train={splits['train']} val={splits['val']} test={splits['test']}")

    print(f"\n  ✓ {total} defect images generated ({len(CLASS_NAMES)} classes × ~{count_per_class})")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    base = Path("/home/claude/pcb_inspection")

    print("=" * 60)
    print("  PCB Synthetic Dataset Generator")
    print("=" * 60)

    # Good boards (Phase 1 unsupervised training)
    print("\n[1/2] Good board images (Phase 1 — PatchCore training)")
    generate_good_images(
        out_dir=str(base / "data/pcb/good"),
        count=300,
    )

    # Defect boards (Phase 2 supervised training)
    print("\n[2/2] Defect images (Phase 2 — YOLOv8 training)")
    generate_defect_images(
        images_dir=str(base / "data/pcb_labeled/images"),
        labels_dir=str(base / "data/pcb_labeled/labels"),
        count_per_class=100,   # 100 × 5 classes = 500 total
    )

    # Summary
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    good_count    = len(list((base / "data/pcb/good").glob("*.jpg")))
    total_defects = sum(
        len(list((base / f"data/pcb_labeled/images/{s}").glob("*.jpg")))
        for s in ("train", "val", "test")
    )
    print(f"  Good images (Phase 1 train) : {good_count}")
    print(f"  Defect images total         : {total_defects}")
    for split in ("train", "val", "test"):
        n = len(list((base / f"data/pcb_labeled/images/{split}").glob("*.jpg")))
        print(f"    {split:6s}: {n}")
    print(f"\n  Ready! Replace images in data/ with real PCB photos when available.")
    print("=" * 60)
