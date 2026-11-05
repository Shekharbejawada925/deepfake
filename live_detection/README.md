# Real-Time Deepfake Detection — Webcam Research Demo

**Status:** Live browser webcam transport is implemented. Legacy ResNeXt50 inference is implemented but requires the user's local checkpoint. Detection accuracy, class mapping, temporal correctness and real-time performance remain **unvalidated**. This is a research prototype, **not** a production authenticity verifier.

## Features
- Browser webcam capture (localhost or HTTPS) with Start/Stop.
- Bounded 8-frame inference batches; only one request in flight per browser tab.
- Flask API with input size, format, count and dimension checks.
- Server-side single-face and blur quality gate, abstention on unusable evidence.
- Native PyTorch legacy checkpoint loader (CPU or optional Apple MPS), loaded once per process.
- Clear status when weights are absent; no fabricated scores or silent fallback.
- API tests, CI, labelled evaluation CLI, corrected batch-first temporal model architecture.

## Run preview (no weights needed)
```bash
git clone https://github.com/Shekharbejawada925/deepfake.git
cd deepfake
git switch feature/live-webcam-detection
cd live_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000/live. Camera works without a checkpoint; detections remain unavailable.

## Run with original ResNeXt50 model
Install supported PyTorch and torchvision for your OS / CPU / Apple Silicon, then:
```bash
pip install -r requirements-model.txt
mkdir -p models
# Put the ORIGINAL state_dict checkpoint at models/df_model.pt locally.
PREDICTOR_MODULE=legacy_model_adapter DEEPFAKE_CHECKPOINT=models/df_model.pt python app.py
```
No weights are included in this repository. If your saved file is named `model_90_acc_60_frames_final_data.pt`, set `DEEPFAKE_CHECKPOINT` to that local filename. The loader expects the exact original state_dict. For Apple Silicon, `DEEPFAKE_USE_MPS=1` optionally enables MPS; benchmark against CPU and keep the faster stable option.

**IMPORTANT:** Legacy model preserves an original LSTM dimension/bias bug. In the one-video inference case it effectively scores the final frame, rather than correctly aggregating time. Thus this compatibility mode should not be advertised as a validated temporal video classifier. Its index mapping 0=FAKE is taken from the original app, not verified from training labels.

## Original project integration
The previous Flask application with /upload and its full templates was **not** in this public repo. This implementation deliberately does not overwrite it. To combine them, register the three live routes in your existing app or use a Flask blueprint, and place `live.html` in its templates folder. Maintain only one server on port 5000.

## Validate before showcasing model quality
1. Verify checkpoint checksum, architecture, input preprocessing and trained class order.
2. Evaluate with `python evaluate.py labelled_manifest.csv --output results.json`; manifest columns `path,label,group`, each path is a directory of eight ordered images.
3. Use independent real/fake videos from identities and sources not seen during training. Report AUROC, F1, accuracy, false positive rate, abstention coverage, confidence calibration and time per inference. The CLI currently computes a confusion matrix, precision, recall, specificity and coverage. Do **not** report any metric without actual ground truth.
4. Train `corrected_model.py` on the labelled training split, then compare it with a mobile/lightweight backbone using the same test protocol.
5. Record inference latency and dropped-frame rate on the actual target hardware before claiming "real time".

## Next optimization paths
- Sample fewer frames / adapt analysis cadence to inference latency.
- Precompute CNN embeddings for overlapping frames rather than re-encoding the whole window.
- Compare lighter face models and CNN backbones experimentally.
- Export a **retrained** model to ONNX and benchmark INT8 quantization against original FP32; quantization can reduce accuracy or speed on some devices.
- Introduce per-user queues and bounded concurrency for server deployment.

## Security
Only use locally. See [SECURITY.md](SECURITY.md) for deployment restrictions and limitations.

## Tests
```bash
python -m unittest discover -s tests -v
```
