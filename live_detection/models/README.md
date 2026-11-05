# Install the supplied checkpoint locally

Weights are deliberately excluded from Git. From `live_detection`, run:

```bash
python -m tools.install_checkpoint /path/to/Deepfake_Streaming_Project_Updated.zip
```

Alternatively pass the original `.pt` file. The installer copies only the verified
checkpoint to `models/legacy_detector.pt`, using the SHA-256 in
`legacy_detector.json`. Keep this profile beside the weights. No download or
retraining is performed. A conflicting checkpoint is never overwritten.

The original graph scores only the final sampled face in a window. Its class
mapping follows the original application and is unverified against training
labels. This baseline does not establish deepfake accuracy.

For a separately trained corrected model, set `MODEL_PATH=models/detector.pt` in
`.env`. See [model setup](../docs/MODEL_SETUP.md). Never relabel a legacy checkpoint
as a corrected model.
