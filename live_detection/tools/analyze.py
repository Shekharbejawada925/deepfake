"""Analyze local videos with the same pipeline used by uploads and live windows."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
from deepfake.inference import Detector, analyze_window, summarize
from deepfake.vision import FaceExtractor, iter_video


def analyze_file(path, detector, max_seconds=300):
    path = Path(path)
    started = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    video = {"fps": cap.get(cv2.CAP_PROP_FPS),
             "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    video["duration_seconds"] = video["frame_count"] / video["fps"] if video["fps"] > 0 else None
    extractor = FaceExtractor(detector.meta)
    pending, timeline, error = [], [], None
    try:
        for sample in iter_video(path, detector.meta["sample_fps"], max_seconds):
            pending.append(sample)
            if len(pending) == detector.meta["sequence_length"]:
                timeline.append(analyze_window(pending, detector, extractor))
                pending = []
        if pending:
            timeline.append(analyze_window(pending, detector, extractor))
    except Exception as exc:
        error = str(exc)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = summarize(timeline)
    if error:
        summary["label"] = "INCONCLUSIVE"
    return {"filename": path.name, "sha256": digest, "ground_truth": None,
            "video": video, "summary": summary, "timeline": timeline,
            "error": error, "runtime_seconds": round(time.perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/legacy_detector.pt"))
    parser.add_argument("--output", type=Path, default=Path("analysis.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve an earlier report.")
    if args.max_seconds < 1 or not all(p.is_file() for p in args.input):
        parser.error("Provide existing video files and a positive duration limit.")
    detector = Detector(args.checkpoint, args.device)
    detector.require_ready()
    report = {"model": detector.status(), "model_metadata": detector.meta,
              "note": "Predictions only. Ground-truth labels were not supplied to this command.", "videos": []}
    for path in args.input:
        result = analyze_file(path, detector, args.max_seconds)
        report["videos"].append(result)
        print(json.dumps({"filename": result["filename"], "summary": result["summary"],
                          "error": result["error"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
