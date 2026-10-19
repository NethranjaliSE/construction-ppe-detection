"""Shared detection logic used by detect_video.py and the Streamlit app."""
from collections import Counter

import cv2
from ultralytics import YOLO

try:
    from src import config as C
except ImportError:
    import config as C

#DEFAULT_WEIGHTS = C.MODELS_DIR / "exp_a_yolo11n.pt"
DEFAULT_WEIGHTS = C.MODELS_DIR / "exp_b_yolo26n.pt"

# Missing hard hat or vest = serious. NO-Mask is shown but not alarmed.
ALERT_CLASSES = {"NO-Hardhat", "NO-Safety Vest"}


def load_model(weights=DEFAULT_WEIGHTS):
    return YOLO(str(weights))


def detect(model, frame, conf=0.25, alert_conf=0.4):
    """Run the model on one image/frame.
    Returns (result, counts per class, list of (class, confidence) alerts)."""
    result = model.predict(frame, conf=conf, verbose=False)[0]
    counts, alerts = Counter(), []
    for cls_id, c in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        name = result.names[int(cls_id)]
        counts[name] += 1
        if name in ALERT_CLASSES and c >= alert_conf:
            alerts.append((name, round(c, 3)))
    return result, dict(counts), alerts


def draw_panel(frame, counts, alerts, fps=None):
    """Counts box (top-left) and a red warning banner (bottom)."""
    lines = ([f"FPS: {fps:.1f}"] if fps else []) + [f"Workers: {counts.get('Person', 0)}"]
    lines += [f"{k}: {v}" for k, v in sorted(counts.items()) if k != "Person"]
    cv2.rectangle(frame, (0, 0), (230, 22 * len(lines) + 10), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if alerts:
        h, w = frame.shape[:2]
        summary = Counter(name for name, _ in alerts)
        msg = "WARNING: " + ", ".join(f"{n} x{k}" for n, k in summary.items())
        cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, msg, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def annotate(result, counts, alerts, fps=None):
    """Boxes + panel. Thinner boxes reduce label clutter in crowded scenes."""
    return draw_panel(result.plot(line_width=2), counts, alerts, fps)