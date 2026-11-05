# Model setup and reproducible evaluation

## Local checkpoint and current behavior

Install the existing checkpoint with `python -m tools.install_checkpoint /path/to/Deepfake_Streaming_Project_Updated.zip`. Weights are excluded from Git.

`models/legacy_detector.pt` is then an unchanged copy of
`model_90_acc_60_frames_final_data(3).pt` supplied by the user.
SHA-256: `57f5daa3bddec309219e0591fbe3d65506fb8d7312b45bffad686695c0089747`.
Safe tensor-only loading and strict shape checks passed for all 322 state entries.
The model has a 2048-unit LSTM with **no bias tensors**, matching the fourth
positional argument bug in the uploaded app. Regression tests check the original and optimized graph equivalence. See `VALIDATION.md` for the checks run on this release.

The original LSTM receives `[1,frames,2048]` but expects sequence-first input.
It therefore reads one timestep with frames as batch items. The selected output
uses the last frame only. `LegacyVideoModel` preserves this graph, including the
unused LeakyReLU and inference-disabled dropout. Encoding only the selected last
frame is mathematically equivalent at batch size one. No weights were retrained.

The sidecar JSON makes assumptions explicit: class order `[fake,real]` comes
from the uploaded app; BGR input and PIL bilinear resize preserve its crop path.
The portable Haar detector replaces the original HOG detector, and strict face
quality checks replace full-frame fallback. Detection/crop distributions can
therefore differ from training. This is an inference baseline with known limits,
not an exact reproduction of the full training pipeline.

Ten-frame reporting windows at 10 fps make live updates practical. Because the
legacy model has no temporal dependence at batch size one, this changes which
frames are reported, not the graph used for each selected face. Nonempty partial
upload windows are allowed in this mode. Corrected temporal models still require
their full trained sequence. Thresholds 0.20/0.80 and temperature 1 are provisional.

To import another copy into a new path (existing output files are preserved):

```bash
python -m tools.import_legacy --input original.pt --output models/another.pt --classes fake,real
```

The importer copies bytes unchanged after tensor/shape checks, writes an explicit
profile, and never infers class semantics from weights. Restart the server after
setting `MODEL_PATH`. Missing/mismatched sidecars are rejected. Do not rename the
checkpoint to a corrected architecture or mark the legacy profile calibrated.

## Remaining inputs for a corrected model

The earlier supplied videos had no authoritative real/fake labels. Personal
clips and their diagnostics are not published in this repository. A filename,
file title, confident softmax value, or
checkpoint name containing “90_acc” does not establish truth or measured accuracy.
Training code, training metadata and representative labeled data are still needed
to verify the class mapping and reduce actual false positives through evaluation.
Merely changing `batch_first`, colors or label strings cannot repair trained weights.

## Prepare a dataset manifest

Obtain authorized copies of FaceForensics++, DFDC and/or Celeb-DF through their
dataset maintainers. No dataset is downloaded automatically. Do not rely on
folder alphabetical order to choose labels.

Create `data/manifest.csv` using `data/manifest.example.csv` as the format. The
example filenames are placeholders and intentionally do not exist. Each row has
`path,label,group,split`. Paths are relative to the manifest directory. Labels
are literally `real` or `fake`; splits are `train`, `val`, or `test`.

Keep every source clip, its manipulated derivatives, and relevant identities in
one group assigned to one split. The tool rejects groups appearing in more than
one split and duplicate paths; it cannot infer an identity from a filename.
Use enough independent subjects and sources in each split. The six-row example
is a format example, not an adequate training dataset.

To follow the paper's nominal 70:30 development/test split on 6,000 videos,
reserve 1,800 videos for untouched testing, then reserve part of the 4,200-video
development pool for validation (for example 840). This yields 3,360 actual
training, 840 validation and 1,800 test videos, subject to identity grouping.
This validation split is an explicit improvement over the paper, which does
not describe an independent calibration set. Never tune on the held-out test set.

## Prepare and train

Run all commands from the project root with its environment active.

Paper-oriented profile: 60-frame windows at 30 sampled fps, first 150 sampled
frames per video. Requires source videos at roughly 30 fps or higher.

```bash
python -m tools.prepare --manifest data/manifest.csv --output data/prepared30 --sequence 60 --fps 30 --max-frames 150
python -m tools.train --prepared data/prepared30 --output models/paper-profile.pt --epochs 15 --batch-size 4
python -m tools.calibrate --prepared data/prepared30 --checkpoint models/paper-profile.pt --output models/paper-calibrated.pt
python -m tools.evaluate --manifest data/manifest.csv --checkpoint models/paper-calibrated.pt --output data/paper-evaluation.json
```

Practical webcam profile: 20-frame windows at 10 fps, approximately two seconds
of collection. This is a different training configuration; do not reuse a
30-fps checkpoint and edit its metadata to 10 fps.

```bash
python -m tools.prepare --manifest data/manifest.csv --output data/prepared10 --sequence 20 --fps 10 --max-frames 150
python -m tools.train --prepared data/prepared10 --output models/webcam-profile.pt --epochs 15 --batch-size 4
python -m tools.calibrate --prepared data/prepared10 --checkpoint models/webcam-profile.pt --output models/detector.pt
python -m tools.evaluate --manifest data/manifest.csv --checkpoint models/detector.pt --output data/webcam-evaluation.json
```

After training, set `MODEL_PATH=models/detector.pt` in `.env` and restart.

Fifteen epochs is a starting training budget, not a promise of convergence. The
best validation-loss checkpoint is saved. Inspect its training log, classwise
errors, abstention and out-of-dataset performance before using it for decisions.
Both classes must remain in train, validation and test after quality filtering.
Review `excluded.json`; report how many sources/windows the detector could not
analyze instead of silently dropping hard cases from evaluation.

The default initializes the CNN using ImageNet weights, which download on the
first training run. Training the large 2048-unit LSTM and CNN benefits from a GPU.
Reduce `--batch-size` if memory is insufficient. `--freeze-cnn` trains a lower
memory baseline with fixed CNN features; this changes the optimization regime
and must be documented. `--from-scratch` disables the ImageNet download and is
not a replacement for deepfake training data.

### Original face detector option

The portable setup uses OpenCV Haar detection. To use the original HOG detector
family, install `requirements-hog.txt`, prepare a new dataset with `--detector hog`,
then train and deploy that checkpoint. The selected detector is recorded in the
checkpoint; the app never switches detectors silently. HOG installation was not
tested in this build; do not edit the bundled legacy detector setting and assume
its results remain equivalent. Haar is less robust for some poses and lighting, so treat
the portable path as a baseline to evaluate.

## Reuse a legacy CNN safely

```bash
python -m tools.init_legacy --input models/legacy_detector.pt --output models/cnn-initialization.pt
python -m tools.train --prepared data/prepared10 --init-cnn models/cnn-initialization.pt --output models/retrained.pt
```

Only the matching ResNeXt encoder tensors are reused. The LSTM and classifier
are newly trained. This avoids preserving the original temporal-layout error,
wrong bias argument and unverified output class order. Only load checkpoints
from a source you trust. The tools use `weights_only=True` and never silently
fall back to arbitrary pickle object loading.

## Calibration and false alarms

Calibration fits a temperature on validation logits and chooses separate
real/fake thresholds using validation error limits. It operates on prepared
windows; final evaluation operates on raw held-out videos and includes windows
that fail quality checks. Validation windows from the same person are correlated;
rates on them are not statistical guarantees on new people or live streams.

The evaluation command reports video-level confusion counts, fake precision,
recall over all fake videos (including abstentions), the false-positive rate
over all real videos, decision coverage, conditional accuracy, accuracy counting
abstentions as undecided, and window processing latency percentiles. Read all
these together. A model that abstains on every difficult video must not be
presented as a high-accuracy detector based only on its easy decided cases.

Evaluate genuine videos with low light, glasses, compression, motion, multiple
faces and absent faces. Include manipulations and cameras not seen during
training. Compare output labels against source truth; never use a prediction
as its own ground-truth label.
