# FrameCheck: Deepfake detection for video streaming

A clean research implementation of the supplied ResNeXt50 + LSTM project, with
video uploads, browser webcam/screen windows, and direct RTSP / HLS / HTTP streams.

**The actual checkpoint loader is integrated.** Git excludes large weights and
personal videos. Install the checkpoint from your saved project ZIP or original
`.pt` file using the command below. Without it, webcam/screen preview works but
predictions remain disabled. You do not need to implement a `PREDICTOR_MODULE`.

The original inference graph uses only the final sampled face of each window.
The bundled compatibility mode preserves that behavior and BGR preprocessing.
It does **not** claim to be a repaired temporal classifier. Class 0=fake and
class 1=real follow the uploaded app; labeled training data were not supplied to
verify that mapping. The paper's 94.3% accuracy has not been reproduced.

A separate corrected RGB / temporal training pipeline is included. Reusing the
old weights under corrected LSTM semantics would silently change their meaning,
so the two model paths are explicit. All predictions are provisional.

## Start on macOS or Linux

Use Python 3.11 or 3.12. Open a terminal inside the extracted `Deepfake_Streaming`
folder (called `live_detection` in the GitHub checkout). An internet connection is needed for initial dependency installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -m tools.install_checkpoint ~/Downloads/Deepfake_Streaming_Project_Updated.zip
python app.py
```

Open **http://127.0.0.1:5000**. If port 5000 is in use on macOS, set `PORT=5050`
in `.env` and open http://127.0.0.1:5050 instead.

For Linux CPU installations, avoid downloading CUDA dependencies by installing
the CPU wheels before the requirements file:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python.exe -m tools.install_checkpoint "$HOME\Downloads\Deepfake_Streaming_Project_Updated.zip"
.venv\Scripts\python.exe app.py
```

After setup, `bash start.sh` or `start.bat` starts the application. The launchers
do not install dependencies automatically. Apple Intel support depends on the
available PyTorch wheels; use a compatible Linux CPU machine if your Python and
Mac architecture do not have wheels for this tested torch/torchvision pair.

## Enable detection

Install from the saved ZIP or pass your original checkpoint directly:

```bash
python -m tools.install_checkpoint /path/to/model_90_acc_60_frames_final_data.pt
```

The installer copies only the expected weights into `models/legacy_detector.pt`,
verifies SHA-256 against `models/legacy_detector.json`, and preserves unrelated
existing weights. It does not retrain, relabel, or download anything. Restart the
server afterward. Keep the neighboring JSON profile. See [Model setup and training](docs/MODEL_SETUP.md)
for the corrected model path. Set `MODEL_PATH=models/detector.pt` only after
training that model.

Command-line analysis uses the exact same window pipeline as the web app:

```bash
python -m tools.analyze --input "my-video.mp4" --output analysis.json
```

Pass several paths after `--input` to analyze a batch. Ground-truth labels are
left unknown; this command reports predictions, not measured accuracy.

Do not swap the REAL/FAKE strings to repair a bad result. Determine the actual
class order, fix preprocessing, retrain the corrected temporal network if
necessary, and evaluate both real and fake videos from unseen subjects.

## Supported input modes

| Mode | Input | Behavior |
| --- | --- | --- |
| Upload | MP4, MOV, AVI, MKV, WebM | Sequential windows; first 300 seconds by default; JSON report |
| Webcam | Browser camera | Sampled JPEG windows, timestamps, cancellation, bounded buffers |
| Screen video | User-selected tab, window or screen | Same model pipeline; browser permission; no audio; Stop sharing ends capture |
| Direct stream | RTSP, HLS playlist or HTTP media stream | Background capture, bounded queue, timeout, cancellation, preview, timeline |

A streaming platform's webpage URL is not a direct video stream. This package
does not bypass DRM, extract YouTube links, inject a browser extension, or connect
directly to Netflix. Use Screen video for visible, capturable browser playback. Codecs depend on the OpenCV FFmpeg build and source format.

### Screen video

Use **Screen video → Share video tab or window** in a supported desktop browser.
Choose the video source in the browser picker. Enlarge the playing video and keep
one face clear. The detector page must stay visible; background-tab timer
throttling can produce irregular windows, which are marked inconclusive. Two
side-by-side windows are useful. Protected/DRM video may appear black and cannot
be analyzed. Nothing bypasses platform permissions or DRM.

Only sampled video frames are sent to this application server. Audio is disabled.
Frames are processed in memory; the JSON export contains scores and timing,
not captured images. Stop session, browser Stop sharing, or the configured
session time limit releases the media tracks. A late permission grant or model
response cannot restart an already stopped capture.

Screen capture needs a supported browser and a secure context, such as localhost
or HTTPS. It begins only after your click and explicit source selection; see the
[screen capture API documentation](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia).

### Direct stream setup

Set the **exact hostname or IP address** in `.env`, restart, and enter the media URL:

```dotenv
STREAM_ALLOWED_HOSTS=192.168.1.20,media.example.org
MAX_STREAM_SECONDS=600
```

Example format: `rtsp://192.168.1.20:8554/live`. Replace it with your actual feed.
Only add sources you control or trust. FFmpeg may follow redirects and HLS
subresources, so this allowlist is an operator opt-in, not a complete network
security boundary. See [Deployment notes](docs/DEPLOYMENT.md).

### Interpreting scores

The UI shows **likely real**, **likely fake** or **inconclusive** per window.
Missing, blurred, small, dark, overexposed, multiple or discontinuous faces receive
no score. A score between the two thresholds is inconclusive. A clip with a fake
window is labeled **flagged segments**, and the timeline identifies its location.
These outcomes refer only to analyzed facial windows. Audio is not analyzed.

Uncalibrated defaults are fake-score ≤ 0.20 for likely real and ≥ 0.80 for likely
fake. These are conservative starting settings, not measured optimal thresholds.
Calibration is a separate step; it cannot guarantee a false-alarm rate on new data.

### Real-time latency

The bundled legacy profile reports every 10 sampled frames at 10 fps, about
one second plus processing and communication. Its score comes from the final
face only. This shorter reporting window preserves the original batch-one
evaluation graph, which has no temporal dependence. Partial upload windows can
also score if their face evidence passes every quality check.

A corrected temporal checkpoint needs a full trained sequence. For example,
20 frames at 10 fps need about two seconds before inference. Changing a corrected
model's sampling rate or sequence length requires preparing/training for that
contract. The UI reads the model settings automatically.

Stream capture continues while inference runs. Old queued samples are discarded
when necessary, the drop count is exposed, and windows containing large gaps
abstain. Sustained real-time throughput is hardware- and workload-dependent.

## Project map

| Path | Purpose |
| --- | --- |
| `app.py` | Local Waitress server entry point |
| `deepfake/model.py` | Corrected temporal network and explicit original-graph compatibility |
| `deepfake/vision.py` | Profile-specific color order, face extraction and fixed-grid sampling |
| `deepfake/inference.py` | Checkpoint loading, quality gates and window scoring |
| `deepfake/streams.py` | Direct stream reader and bounded capture queue |
| `deepfake/jobs.py` | Upload/stream background jobs, cleanup and cancellation |
| `deepfake/web.py` | Flask API, sessions, input limits and private reports |
| `templates/`, `static/` | Responsive, self-contained web interface |
| `tools/` | Preparation, training, calibration, legacy CNN extraction and evaluation |
| `tests/` | Regression, API, model-shape and stream-lifecycle tests |
| `docs/` | Code audit, paper mapping, setup, API, deployment and validation |

## Run the tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
node --test tests/live-capture.test.mjs
```

Tests use generated media and explicit test doubles for predictable API results.
The real neural network is tested for tensor shape, temporal influence and batch
independence with random weights. **Passing these tests does not establish
deepfake accuracy.** See `docs/VALIDATION.md` for the checks performed for this build.

## Docker (optional; CPU)

```bash
docker build -t framecheck .
docker run --rm -p 127.0.0.1:5000:5000 --env-file .env -e HOST=0.0.0.0 -v "$(pwd)/models:/app/models:ro" framecheck
```

Docker commands are provided for deployment convenience; the image was not built
in the validation environment. Local Python execution is the tested route.

Read [Audit](docs/CODE_AUDIT.md), [Paper alignment](docs/PAPER_ALIGNMENT.md),
[Model setup](docs/MODEL_SETUP.md), and [API](docs/API.md) before extending the system.
