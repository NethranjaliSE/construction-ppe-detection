"""Dataset loading, inspection and preparation."""
from collections import Counter
from pathlib import Path

import hashlib
import cv2
import shutil
import pandas as pd
import yaml
import re


try:
    from src import config as C
except ImportError:
    import config as C


def load_class_names(dataset_dir=C.DATASET_DIR):
    """Read the class names from the dataset's data.yaml."""
    with open(dataset_dir / "data.yaml") as f:
        names = yaml.safe_load(f)["names"]
    return list(names)


def count_objects_per_class(names, dataset_dir=C.DATASET_DIR):
    """Count labelled objects (boxes) per class in each split."""
    table = {}
    for split in C.SPLITS:
        counts = Counter()
        for label_file in (dataset_dir / split / "labels").glob("*.txt"):
            for line in label_file.read_text().splitlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    counts[class_id] += 1
        table[split] = [counts.get(i, 0) for i in range(len(names))]

    df = pd.DataFrame(table, index=names)
    df["total"] = df.sum(axis=1)
    return df.sort_values("total", ascending=False)


def remap_dataset(names, id_map, src_dir=C.DATASET_DIR, out_dir=C.PROCESSED_DIR):
    """Copy images and write remapped labels to out_dir. The raw data is never changed."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    kept = dropped = 0
    for split in C.SPLITS:
        shutil.copytree(src_dir / split / "images", out_dir / split / "images")
        (out_dir / split / "labels").mkdir(parents=True)
        for label_file in (src_dir / split / "labels").glob("*.txt"):
            new_lines = []
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                new_id = id_map[int(parts[0])]
                if new_id is None:
                    dropped += 1
                    continue
                new_lines.append(" ".join([str(new_id)] + parts[1:]))
                kept += 1
            (out_dir / split / "labels" / label_file.name).write_text("\n".join(new_lines))
    print(f"Boxes kept: {kept} | dropped: {dropped}")


def write_data_yaml(root, names, out_path=C.DATA_YAML):
    """data.yaml with an absolute path, so training works from any folder."""
    cfg = {"path": str(root.resolve()), "train": "train/images",
           "val": "valid/images", "test": "test/images",
           "names": {i: n for i, n in enumerate(names)}}
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("Wrote", out_path)


def inspect_images(dataset_dir=C.PROCESSED_DIR):
    """One row per image: can it be opened, its size, brightness, and number of boxes."""
    rows = []
    for split in C.SPLITS:
        for img_path in sorted((dataset_dir / split / "images").glob("*")):
            img = cv2.imread(str(img_path))
            label = dataset_dir / split / "labels" / f"{img_path.stem}.txt"
            n_boxes = len(label.read_text().split("\n")) if label.exists() and label.read_text().strip() else 0
            row = {"split": split, "file": img_path.name, "readable": img is not None,
                   "has_label": label.exists(), "n_boxes": n_boxes}
            if img is not None:
                row["height"], row["width"] = img.shape[:2]
                row["brightness"] = img.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def validate_labels(num_classes, dataset_dir=C.PROCESSED_DIR):
    """Check every label line against the YOLO format rules from the brief."""
    issues = []
    for split in C.SPLITS:
        for label_file in (dataset_dir / split / "labels").glob("*.txt"):
            for line_no, line in enumerate(label_file.read_text().splitlines(), 1):
                parts = line.split()
                problem = None
                if len(parts) != 5:
                    problem = f"{len(parts)} values instead of 5"
                else:
                    cid, x, y, w, h = int(parts[0]), *map(float, parts[1:])
                    if not 0 <= cid < num_classes:
                        problem = f"class id {cid} out of range"
                    elif not all(0 <= v <= 1 for v in (x, y, w, h)):
                        problem = "coordinate not between 0 and 1"
                    elif w <= 0 or h <= 0:
                        problem = "width or height not positive"
                    elif x - w/2 < -0.001 or x + w/2 > 1.001 or y - h/2 < -0.001 or y + h/2 > 1.001:
                        problem = "box goes outside the image"
                if problem:
                    issues.append({"split": split, "file": label_file.name,
                                   "line": line_no, "problem": problem})
    return pd.DataFrame(issues, columns=["split", "file", "line", "problem"])


def read_boxes(label_path):
    """List of (class_id, x, y, w, h) from one label file."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes

def find_cross_split_duplicates(dataset_dir=C.PROCESSED_DIR, max_distance=3):
    """Find identical or near-identical images that appear in different splits."""
    def image_hash(img):
        # 'difference hash': shrink to 9x8 grey, compare neighbouring pixels -> 64 bits
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8))
        return (small[:, 1:] > small[:, :-1]).flatten()

    rows = []
    for split in C.SPLITS:
        for p in sorted((dataset_dir / split / "images").glob("*")):
            img = cv2.imread(str(p))
            rows.append({"split": split, "file": p.name,
                         "source": p.stem.split(".rf.")[0],          # Roboflow original name
                         "md5": hashlib.md5(p.read_bytes()).hexdigest(),
                         "hash": image_hash(img)})
    df = pd.DataFrame(rows)

    found = []
    # 1) same original file name in two splits
    for src, g in df.groupby("source"):
        if g["split"].nunique() > 1:
            found.append({"type": "same source name", "files": ", ".join(g["file"]),
                          "splits": ", ".join(g["split"])})
    # 2) near-identical pictures in two splits (works even if renamed or re-saved)
    hashes, splits = list(df["hash"]), list(df["split"])
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if splits[i] != splits[j] and (hashes[i] != hashes[j]).sum() <= max_distance:
                found.append({"type": "near-duplicate", "files": f"{df.file[i]} | {df.file[j]}",
                              "splits": f"{splits[i]}, {splits[j]}"})
    return pd.DataFrame(found, columns=["type", "files", "splits"])

def group_key(filename):
    """Original source of an image. Frames of one video share a key:
    'IMG_0871_mp4-23_jpg.rf.abc.jpg' -> 'IMG_0871_mp4'."""
    stem = filename.split(".rf.")[0]
    m = re.match(r"(.+_(?:mp4|mov))-\d+_jpg$", stem, re.IGNORECASE)
    return m.group(1) if m else stem


def move_leaked_to_train(leaks, dataset_dir=C.PROCESSED_DIR):
    """Move valid/test images that leak from train into train, so test stays unseen."""
    train_keys = {group_key(p.name) for p in (dataset_dir / "train" / "images").glob("*")}
    to_move = set()

    # 1) same source or same video as a train image
    for split in ["valid", "test"]:
        for p in (dataset_dir / split / "images").glob("*"):
            if group_key(p.name) in train_keys:
                to_move.add((split, p.name))

    # 2) near-duplicate pictures found by the image hash
    for _, row in leaks[leaks["type"] == "near-duplicate"].iterrows():
        files = [f.strip() for f in row["files"].split("|")]
        splits = [s.strip() for s in row["splits"].split(",")]
        for f, s in zip(files, splits):
            if s != "train":
                to_move.add((s, f))

    for split, name in sorted(to_move):
        shutil.move(dataset_dir / split / "images" / name, dataset_dir / "train" / "images" / name)
        label = dataset_dir / split / "labels" / f"{Path(name).stem}.txt"
        if label.exists():
            shutil.move(label, dataset_dir / "train" / "labels" / label.name)

    print(f"Moved {len(to_move)} images into train")
    return pd.DataFrame(sorted(to_move), columns=["moved_from", "file"])

if __name__ == "__main__":
    try:
        from src.feature_engineering import TARGET_CLASSES, build_id_map
    except ImportError:
        from feature_engineering import TARGET_CLASSES, build_id_map

    names = load_class_names()
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    count_objects_per_class(names).to_csv(C.METRICS_DIR / "class_distribution_raw.csv")

    id_map = build_id_map(names)
    remap_dataset(names, id_map)
    leaks = find_cross_split_duplicates()
    moved = move_leaked_to_train(leaks)
    moved.to_csv(C.METRICS_DIR / "leakage_fix_moved_files.csv", index=False)
    write_data_yaml(C.PROCESSED_DIR, TARGET_CLASSES)

    after = count_objects_per_class(TARGET_CLASSES, C.PROCESSED_DIR)
    print(after)
    after.to_csv(C.METRICS_DIR / "class_distribution_final.csv")