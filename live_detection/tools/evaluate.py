"""Evaluate raw held-out videos; include abstention, false alarms and latency."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from deepfake.inference import Detector, analyze_window, summarize
from deepfake.vision import FaceExtractor, iter_video
from .dataset import read_manifest


def evaluate_rows(rows,detector,max_seconds):
    records, latencies, window_scores = [], [], []
    extractor = FaceExtractor(detector.meta)
    for row in rows:
        windows, pending, error = [], [], None
        started = time.perf_counter()
        try:
            for sample in iter_video(row["path"],detector.meta["sample_fps"],max_seconds):
                pending.append(sample)
                if len(pending) == detector.meta["sequence_length"]:
                    windows.append(analyze_window(pending,detector,extractor)); pending=[]
            if pending:
                windows.append(analyze_window(pending,detector,extractor))
        except Exception as exc:
            error = str(exc)
        summary = summarize(windows)
        prediction = ("fake" if summary["label"] == "FLAGGED SEGMENTS" else
                      "real" if summary["label"] == "LIKELY REAL" else "abstain")
        if error:
            prediction = "abstain"
        records.append({"path":row["path"],"truth":row["label"],"prediction":prediction,
                        "summary":summary,"error":error,"elapsed_seconds":time.perf_counter()-started})
        for result in windows:
            latencies.append(result["processing_ms"])
            if result["p_fake"] is not None:
                window_scores.append({"truth":row["label"],"p_fake":result["p_fake"]})
    scored = [r for r in records if r["prediction"] != "abstain"]
    tp = sum(r["truth"] == "fake" and r["prediction"] == "fake" for r in records)
    fp = sum(r["truth"] == "real" and r["prediction"] == "fake" for r in records)
    tn = sum(r["truth"] == "real" and r["prediction"] == "real" for r in records)
    fn = sum(r["truth"] == "fake" and r["prediction"] == "real" for r in records)
    real_n, fake_n = (sum(r["truth"] == label for r in records) for label in ("real","fake"))
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/fake_n if fake_n else None  # abstentions on fake videos count as undetected
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.
    metrics = {"videos":len(records),"decided_videos":len(scored),"abstained_videos":len(records)-len(scored),
               "coverage":len(scored)/len(records) if records else 0,
               "accuracy_on_decided_videos":(tp+tn)/len(scored) if scored else None,
               "correct_decisions_over_all_videos":(tp+tn)/len(records) if records else None,
               "fake_precision":precision,"fake_recall_over_all_fake_videos":recall,"f1_including_missed_abstentions":f1,
               "false_positive_rate_over_all_real_videos":fp/real_n if real_n else None,
               "confusion":{"true_fake":tp,"false_fake":fp,"true_real":tn,"false_real":fn},
               "p50_window_processing_ms":float(np.percentile(latencies,50)) if latencies else None,
               "p95_window_processing_ms":float(np.percentile(latencies,95)) if latencies else None,
               "window_collection_seconds":detector.meta["sequence_length"]/detector.meta["sample_fps"],
               "scored_windows":len(window_scores)}
    return {"metrics":metrics,"videos":records,"window_scores":window_scores,"model":detector.status()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",default="evaluation.json")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--max-seconds",type=int,default=300)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a new report path.")
    detector = Detector(args.checkpoint,args.device); detector.require_ready()
    rows = read_manifest(args.manifest,required_splits=("test",))
    rows = [r for r in rows if r["split"] == "test"]
    used = set(detector.meta.get("training",{}).get("train_groups",[])) | set(detector.meta.get("training",{}).get("val_groups",[])) | set(detector.meta.get("calibration",{}).get("groups",[]))
    if used & {r["group"] for r in rows}:
        raise ValueError("Test groups overlap training or calibration subjects.")
    result = evaluate_rows(rows,detector,args.max_seconds)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["metrics"],indent=2))


if __name__ == "__main__":
    main()
