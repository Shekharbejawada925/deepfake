"""Extract only compatible CNN weights from a trusted legacy state_dict."""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists. Choose a new file.")
    payload = torch.load(args.input,map_location="cpu",weights_only=True)
    if not isinstance(payload,dict):
        raise ValueError("Expected a tensor state dictionary.")
    state = payload.get("state_dict",payload)
    cleaned = {key.removeprefix("module."):value for key,value in state.items()}
    cnn = {key:value for key,value in cleaned.items() if key.startswith("model.") and torch.is_tensor(value)}
    if not cnn:
        raise ValueError("No ResNeXt encoder weights were found with the model. prefix.")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":cnn,"initialization_only":True,
                "note":"CNN initialization only; LSTM and classifier require retraining."},args.output)
    print("Saved CNN initialization. This file cannot be used as a detector checkpoint.")


if __name__ == "__main__":
    main()
