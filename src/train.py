"""Baseline, training and evaluation.

python src/train.py train --epochs 2 --fraction 0.2 --name smoke_test   # quick test
python src/train.py baseline                                              # Phase 5
python src/train.py train --model yolo11n.pt --name exp_a_yolo11n         # Phase 6
python src/train.py evaluate --name exp_a_yolo11n --split test            # Phase 7
"""
import argparse
import json
import shutil

import pandas as pd
from ultralytics import YOLO

try:
    from src import config as C
except ImportError:
    import config as C

RUNS_DIR = C.MODELS_DIR / "runs"


def save_metrics(tag, m):
    """Save overall + per-class metrics from a validation result."""
    overall = {"precision": m.box.mp, "recall": m.box.mr,
               "mAP50": m.box.map50, "mAP50_95": m.box.map,
               "inference_ms": m.speed["inference"]}
    per_class = []
    for i, cid in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class.append({"class": m.names[int(cid)], "precision": p, "recall": r,
                          "mAP50": ap50, "mAP50_95": ap})
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (C.METRICS_DIR / f"{tag}.json").write_text(json.dumps(overall, indent=2, default=float))
    pd.DataFrame(per_class).round(4).to_csv(C.METRICS_DIR / f"{tag}_per_class.csv", index=False)
    print(f"\n{tag}: " + "  ".join(f"{k}={v:.3f}" for k, v in overall.items()))
    return overall


def baseline(args):
    """Phase 5: pretrained COCO model before any fine-tuning."""
    model = YOLO(args.model)
    m = model.val(data=str(C.DATA_YAML), split="val", imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project=str(RUNS_DIR), name="baseline", exist_ok=True)
    save_metrics("baseline_val", m)


def train(args):
    """Phase 6: fine-tune pretrained weights on our 10 classes."""
    model = YOLO(args.model)
    model.train(data=str(C.DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, seed=C.SEED, device=args.device, workers=args.workers,
                fraction=args.fraction, project=str(RUNS_DIR), name=args.name, exist_ok=True,
                plots=True)
    best = RUNS_DIR / args.name / "weights" / "best.pt"
    shutil.copy(best, C.MODELS_DIR / f"{args.name}.pt")
    print("Best weights ->", C.MODELS_DIR / f"{args.name}.pt")
    args.split = "val"
    overall = evaluate(args)

    row = {"name": args.name, "model": args.model, "imgsz": args.imgsz,
           "epochs": args.epochs, **{k: round(float(v), 4) for k, v in overall.items()}}
    log = C.METRICS_DIR / "experiments.csv"
    df = pd.read_csv(log) if log.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df["name"] != args.name]
    pd.concat([df, pd.DataFrame([row])]).to_csv(log, index=False)


def evaluate(args):
    """Phase 7: metrics on the validation set or (once, at the end) the test set."""
    model = YOLO(str(C.MODELS_DIR / f"{args.name}.pt"))
    m = model.val(data=str(C.DATA_YAML), split=args.split, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project=str(RUNS_DIR), name=f"{args.name}_{args.split}",
                  exist_ok=True, plots=True)
    return save_metrics(f"{args.name}_{args.split}", m)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "train", "evaluate"])
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--name", default="exp_a_yolo11n")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()
    {"baseline": baseline, "train": train, "evaluate": evaluate}[args.mode](args)