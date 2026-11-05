"""Create valid, chronological RGB face windows using inference preprocessing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from deepfake.contract import default_metadata, LENGTHS
from deepfake.vision import FaceExtractor, iter_video
from deepfake.inference import prepare_window
from .dataset import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--sequence",type=int,choices=LENGTHS,default=60)
    parser.add_argument("--fps",type=float,default=30)
    parser.add_argument("--max-frames",type=int,default=150,help="Maximum sampled frames per source video")
    parser.add_argument("--detector",choices=("haar","hog"),default="haar")
    args = parser.parse_args()
    if not 1 <= args.fps <= 30 or args.max_frames < args.sequence:
        parser.error("fps must be 1..30 and max-frames must cover a full window")
    rows = read_manifest(args.manifest,required_splits=("train","val","test"))
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory to avoid stale or mixed preprocessing.")
    output.mkdir(parents=True,exist_ok=True)
    meta = default_metadata(args.sequence,args.fps,args.detector)
    extractor = FaceExtractor(meta)
    prepared, skipped = [], []
    for row in rows:
        samples, window_no = [], 0
        try:
            iterator = iter_video(row["path"],args.fps,max_seconds=300)
            try:
                for index, sample in enumerate(iterator):
                    if index >= args.max_frames:
                        break
                    samples.append(sample)
                    if len(samples) == args.sequence:
                        crops, _, reasons = prepare_window(samples,extractor,meta)
                        if reasons:
                            skipped.append({"path":row["path"],"window":window_no,"reasons":reasons})
                        else:
                            key = hashlib.sha256((row["path"]+":"+str(window_no)).encode()).hexdigest()[:24]
                            target = output/(key+".npz")
                            np.savez_compressed(target,frames=np.stack(crops).astype(np.uint8))
                            prepared.append({**row,"path":target.name,"source":row["path"]})
                        window_no += 1
                        samples = []
            finally:
                iterator.close()
            if samples:
                skipped.append({"path":row["path"],"window":window_no,"reasons":{"insufficient_frames":len(samples)}})
        except Exception as exc:
            skipped.append({"path":row["path"],"error":str(exc)})
        print(f"Prepared {len(prepared)} windows; excluded {len(skipped)} windows/sources",flush=True)
    with (output/"manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=("path","label","group","split","source"))
        writer.writeheader(); writer.writerows([{k:r[k] for k in writer.fieldnames} for r in prepared])
    (output/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    (output/"excluded.json").write_text(json.dumps(skipped,indent=2),encoding="utf-8")
    if not prepared:
        raise SystemExit("No valid windows. Check video duration, frame rate, face size and lighting.")


if __name__ == "__main__":
    main()
