"""Copy a legacy state dict unchanged and record explicit inference assumptions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from deepfake.contract import legacy_metadata, validate_metadata
from deepfake.model import LegacyVideoModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/legacy_detector.pt"))
    parser.add_argument("--classes", required=True, choices=["fake,real", "real,fake"],
                        help="Class order assumed from the original app, not inferred from weights")
    args = parser.parse_args()
    sidecar = args.output.with_suffix(".json")
    if args.output.exists() or sidecar.exists():
        parser.error("Output already exists. Choose a new path to preserve it.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    state = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
        isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()
    ):
        parser.error("Expected a finite tensor-only raw state dictionary.")
    model = LegacyVideoModel()
    model.load_state_dict(state, strict=True)
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    meta = legacy_metadata(digest)
    meta["classes"] = args.classes.split(",")
    meta["label_mapping_source"] = (f"Explicit import assumption: index 0={meta['classes'][0]}, "
                                    f"index 1={meta['classes'][1]}; training labels unavailable")
    validate_metadata(meta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, args.output)
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "profile": str(sidecar),
                      "sha256": digest, "tensor_count": len(state),
                      "warning": "Original eval graph only. Training labels and accuracy are unverified."}, indent=2))


if __name__ == "__main__":
    main()
