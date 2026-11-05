import csv
from pathlib import Path


def read_manifest(path, required_splits=None):
    path = Path(path).resolve()
    with path.open(newline="",encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"path","label","group","split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs path,label,group,split columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest has no videos.")
    groups, paths = {}, set()
    for row in rows:
        row["label"] = row["label"].strip().lower()
        row["split"] = row["split"].strip().lower()
        row["group"] = row["group"].strip()
        video = Path(row["path"])
        video = (path.parent / video).resolve() if not video.is_absolute() else video.resolve()
        row["path"] = str(video)
        if row["label"] not in ("real","fake") or row["split"] not in ("train","val","test") or not row["group"]:
            raise ValueError("Every row requires real/fake, train/val/test, and a source/identity group.")
        if not video.is_file():
            raise ValueError("Video or prepared sample not found: " + str(video))
        if video in paths:
            raise ValueError("Duplicate video/sample path: " + str(video))
        paths.add(video)
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("Data leakage: the same source/identity group appears in multiple splits: " + row["group"])
        groups[row["group"]] = row["split"]
    for split in required_splits or []:
        if {r["label"] for r in rows if r["split"] == split} != {"real","fake"}:
            raise ValueError(f"Split {split} must contain real and fake videos.")
    return rows
