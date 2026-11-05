"""Train the corrected temporal model on prepared, group-separated windows."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from deepfake.contract import validate_metadata, ARCHITECTURE
from deepfake.model import VideoModel, select_device
from deepfake.vision import normalize_crops
from .dataset import read_manifest


class Windows(Dataset):
    def __init__(self, rows, meta):
        self.rows, self.meta = rows, meta
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,index):
        row = self.rows[index]
        with np.load(row["path"],allow_pickle=False) as data:
            frames = data["frames"]
        if frames.shape != (self.meta["sequence_length"],112,112,3):
            raise ValueError("Prepared window shape disagrees with metadata.")
        return torch.from_numpy(normalize_crops(frames,self.meta)), self.meta["classes"].index(row["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",required=True)
    parser.add_argument("--output",default="models/detector.pt")
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--lr",type=float,default=0.0001)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--workers",type=int,default=0)
    parser.add_argument("--init-cnn",help="Output of tools.init_legacy")
    parser.add_argument("--from-scratch",action="store_true",help="No ImageNet download; usually needs more training data")
    parser.add_argument("--freeze-cnn",action="store_true",help="Lower memory baseline; keep CNN and its BN statistics frozen")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    if Path(args.output).exists():
        parser.error("Choose a new output path to preserve the previous model.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    prepared = Path(args.prepared)
    meta = validate_metadata(json.loads((prepared/"metadata.json").read_text()))
    if meta["architecture"] != ARCHITECTURE:
        raise ValueError("Train only from freshly prepared RGB data for the corrected architecture.")
    rows = read_manifest(prepared/"manifest.csv",required_splits=("train","val","test"))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    loaders = {"train":DataLoader(Windows(train,meta),batch_size=args.batch_size,shuffle=True,num_workers=args.workers),
               "val":DataLoader(Windows(val,meta),batch_size=args.batch_size,shuffle=False,num_workers=args.workers)}
    device = select_device(args.device)
    model = VideoModel(pretrained=not(args.from_scratch or args.init_cnn)).to(device)
    if args.init_cnn:
        payload = torch.load(args.init_cnn,map_location="cpu",weights_only=True)
        if payload.get("initialization_only") is not True:
            raise ValueError("Use tools.init_legacy to create a CNN initialization first.")
        state = payload["state_dict"]
        expected = {k for k in model.state_dict() if k.startswith("model.")}
        if set(state) != expected:
            raise ValueError("Legacy encoder architecture does not match ResNeXt50_32x4d.")
        model.load_state_dict(state,strict=False)
    if args.freeze_cnn:
        for parameter in model.model.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-4)
    counts = [sum(r["label"] == label for r in train) for label in meta["classes"]]
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([len(train)/(2*n) for n in counts],dtype=torch.float32,device=device))
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    best_loss, log = float("inf"), []
    for epoch in range(args.epochs):
        record = {"epoch":epoch+1}
        for split,loader in loaders.items():
            training = split == "train"
            model.train(training)
            if args.freeze_cnn:
                model.model.eval()
            loss_total, correct, seen = 0., 0, 0
            for inputs,labels in loader:
                inputs,labels = inputs.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    logits = model(inputs)
                    loss = criterion(logits,labels)
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                        optimizer.step()
                seen += len(labels); loss_total += float(loss.item())*len(labels)
                correct += int((logits.argmax(1) == labels).sum().item())
            record[split+"_loss"] = loss_total/seen
            record[split+"_accuracy"] = correct/seen
        log.append(record); print(json.dumps(record),flush=True)
        if record["val_loss"] < best_loss:
            best_loss = record["val_loss"]
            meta.update(training_complete=True,calibrated=False)
            meta["training"] = {"best_epoch":epoch+1,"seed":args.seed,"train_windows":len(train),"val_windows":len(val),
                                "manifest_sha256":hashlib.sha256((prepared/"manifest.csv").read_bytes()).hexdigest(),
                                "cnn_frozen":args.freeze_cnn,
                                "train_groups":sorted({r["group"] for r in train}),
                                "val_groups":sorted({r["group"] for r in val})}
            torch.save({"metadata":meta,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},output)
        output.with_suffix(".training.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    print("Saved best validation-loss checkpoint. Next calibrate on val, then evaluate on untouched test videos.")


if __name__ == "__main__":
    main()
