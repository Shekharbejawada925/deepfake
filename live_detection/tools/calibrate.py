"""Fit temperature and provisional decision thresholds using validation data only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from deepfake.contract import validate_metadata, ARCHITECTURE
from deepfake.model import VideoModel, select_device
from .dataset import read_manifest
from .train import Windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",default="models/detector-calibrated.pt")
    parser.add_argument("--max-false-positive-rate",type=float,default=0.05)
    parser.add_argument("--device",default="auto")
    args = parser.parse_args()
    if not 0 < args.max_false_positive_rate < 0.5:
        parser.error("max-false-positive-rate must lie between 0 and 0.5")
    if Path(args.output).exists():
        parser.error("Choose a new output checkpoint.")
    payload = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if "metadata" not in payload or "state_dict" not in payload:
        raise ValueError("Calibrate a corrected, trained checkpoint; the raw legacy model is an unvalidated compatibility baseline.")
    meta = validate_metadata(payload["metadata"])
    if meta["architecture"] != ARCHITECTURE:
        raise ValueError("Calibration tool requires the corrected architecture.")
    if not meta["training_complete"]:
        raise ValueError("Train the detector first.")
    prepared = Path(args.prepared)
    contract = validate_metadata(json.loads((prepared/"metadata.json").read_text()))
    for key in ("sequence_length","sample_fps","detector","crop_margin","min_face_pixels","min_blur","classes"):
        if contract[key] != meta[key]:
            raise ValueError("Prepared validation data does not match checkpoint: "+key)
    rows = read_manifest(prepared/"manifest.csv",required_splits=("val",))
    rows = [r for r in rows if r["split"] == "val"]
    trained_groups = set(meta.get("training",{}).get("train_groups",[]))
    if trained_groups & {r["group"] for r in rows}:
        raise ValueError("Calibration groups overlap training groups.")
    device = select_device(args.device)
    model = VideoModel().to(device); model.load_state_dict(payload["state_dict"],strict=True); model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for x,y in DataLoader(Windows(rows,meta),batch_size=1):
            logits.append(model(x.to(device)).cpu()); targets.append(y)
    logits, targets = torch.cat(logits), torch.cat(targets)
    log_t = torch.zeros((),requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t],lr=0.1,max_iter=50)
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits/log_t.exp().clamp(0.05,10),targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_t.detach().exp().clamp(0.05,10))
    p = torch.softmax(logits/temperature,dim=1)[:,meta["classes"].index("fake")].numpy()
    y = (targets.numpy() == meta["classes"].index("fake")).astype(int)
    candidates_fake = [float(t) for t in np.linspace(.51,.999,200) if np.mean(p[y==0] >= t) <= args.max_false_positive_rate]
    candidates_real = [float(t) for t in np.linspace(.001,.49,200) if np.mean(p[y==1] <= t) <= args.max_false_positive_rate]
    if not candidates_fake or not candidates_real:
        raise ValueError("No suitable uncertainty thresholds meet the requested validation error limits. Improve the model.")
    meta.update(temperature=temperature,real_threshold=max(candidates_real),fake_threshold=min(candidates_fake),calibrated=True)
    meta["calibration"] = {"split":"val","windows":len(rows),"requested_error_rate":args.max_false_positive_rate,
                           "groups":sorted({r["group"] for r in rows}),
                           "note":"In-sample validation rates are not guarantees on new videos or test subjects."}
    validate_metadata(meta)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"metadata":meta,"state_dict":payload["state_dict"]},args.output)
    print(json.dumps(meta["calibration"],indent=2))


if __name__ == "__main__":
    main()
