"""Real-time PPE detection on a video file, webcam or single image.

Examples (run from the project root):
    python src/detect_video.py --source samples/site1.mp4
    python src/detect_video.py --source 0                     # webcam
    python src/detect_video.py --source samples/photo.jpg     # one image
    python src/detect_video.py --source samples/site1.mp4 --weights models/exp_b_yolo26n.pt

Press Q in the window to stop.
Outputs go to reports/predictions/run_<date>_<time>/:
    output.mp4       annotated video (or <name>_annotated.jpg for an image)
    alerts/          snapshots of frames with violations (max one every 2 s)
    violations.csv   every violation with frame number, video time and confidence
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    from src import config as C
    from src.model import DEFAULT_WEIGHTS, load_model, detect, annotate
except ImportError:              # when run as "python src/detect_video.py"
    import config as C
    from model import DEFAULT_WEIGHTS, load_model, detect, annotate

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_image(model, path, out_dir, conf, alert_conf):
    """Detect on one image, save it, and show it until a key is pressed."""
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")

    result, counts, alerts = detect(model, frame, conf, alert_conf)
    annotated = annotate(result, counts, alerts)

    out_path = out_dir / f"{Path(path).stem}_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    print("Counts :", counts)
    print("Alerts :", alerts if alerts else "none")
    print("Saved  :", out_path)

    cv2.imshow("PPE Safety Monitor (press any key to close)", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(model, source, out_dir, conf, alert_conf, show=True):
    """Detect frame by frame on a video file or webcam."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25          # webcams often report 0
    (out_dir / "alerts").mkdir(parents=True, exist_ok=True)

    log_file = open(out_dir / "violations.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["frame", "video_time_s", "class", "confidence"])

    writer = None
    frame_no, fps, last_alert_frame = 0, 0.0, -10**9
    total_alerts = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # --- detection (timed: model latency only) ---
            t0 = time.perf_counter()
            result, counts, alerts = detect(model, frame, conf, alert_conf)
            dt = max(time.perf_counter() - t0, 1e-6)
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt   # smoothed FPS

            # --- log every violation ---
            for name, c in alerts:
                log.writerow([frame_no, round(frame_no / fps_in, 2), name, c])
            total_alerts += len(alerts)

            annotated = annotate(result, counts, alerts, fps)

            # --- save at most one alert snapshot every 2 seconds of video ---
            if alerts and frame_no - last_alert_frame >= 2 * fps_in:
                cv2.imwrite(str(out_dir / "alerts" / f"frame_{frame_no:05d}.jpg"), annotated)
                last_alert_frame = frame_no

            # --- write annotated video ---
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "output.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
            writer.write(annotated)

            if show:
                cv2.imshow("PPE Safety Monitor (press Q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_no += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        log_file.close()
        cv2.destroyAllWindows()

    print(f"Processed {frame_no} frames | avg model FPS ~{fps:.1f} | "
          f"violations logged: {total_alerts}")
    print("Outputs:", out_dir)


def main():
    ap = argparse.ArgumentParser(description="PPE safety detection on video, webcam or image")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="trained .pt model")
    ap.add_argument("--source", required=True, help="video path, image path, or 0 for webcam")
    ap.add_argument("--conf", type=float, default=0.25, help="draw boxes above this confidence")
    ap.add_argument("--alert-conf", type=float, default=0.4, help="raise a warning above this confidence")
    ap.add_argument("--no-show", action="store_true", help="don't open a window (faster)")
    args = ap.parse_args()

    model = load_model(args.weights)
    out_dir = C.REPORTS_DIR / "predictions" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source.isdigit():                                   # webcam
        run_video(model, int(args.source), out_dir, args.conf, args.alert_conf, not args.no_show)
    elif Path(args.source).suffix.lower() in IMAGE_EXTS:        # single image
        run_image(model, args.source, out_dir, args.conf, args.alert_conf)
    else:                                                       # video file
        run_video(model, args.source, out_dir, args.conf, args.alert_conf, not args.no_show)


if __name__ == "__main__":
    main()