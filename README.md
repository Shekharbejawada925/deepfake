# Deepfake detection for video streaming

Live facial-video analysis with Flask, OpenCV, PyTorch, and a ResNeXt50 + LSTM model.

The application supports **webcam**, **shared-screen video**, **direct RTSP / HLS / HTTP streams**, and **uploaded videos**. It reports likely real, likely fake, or inconclusive windows, with a score timeline and a downloadable JSON report.

## Run locally

Use Python 3.11 or 3.12. From this repository:

```bash
cd live_detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m tools.install_checkpoint ~/Downloads/Deepfake_Streaming_Project_Updated.zip
python app.py
```

Open **http://127.0.0.1:5000/live**. See [full setup and Windows commands](live_detection/README.md).

The installer uses the checkpoint from your previously saved project ZIP; it does not download a model. You can also pass the original `.pt` file. Large weights and personal videos are excluded from Git. Without weights, camera and screen preview still work, but predictions stay disabled.

## Check live video

1. **Webcam:** click Start webcam and allow camera access.
2. **Screen video:** play a video in another tab/window, click Share video tab or window, and choose that source in the browser picker. Enlarge the video so one face is clear. Keep the analysis page visible beside it.
3. **Stream URL:** configure the trusted media host in `.env`, restart, and paste a direct media URL. See [stream setup](live_detection/README.md#direct-stream-setup).

Stop session releases browser capture and ignores pending results. Browser Stop sharing also stops analysis. Frames from browser capture are processed in memory; audio is not captured or analyzed.

## What the model can establish

The supplied checkpoint is supported through an explicit compatibility mode. Its original LSTM layout uses only the final sampled face in each window. Its class mapping and decision thresholds have not been validated on labeled data. A high score is **not proof of authenticity or measured accuracy**.

The corrected temporal architecture and preparation, training, calibration, and evaluation commands are also included. Corrected temporal inference requires retraining; the existing weights cannot be silently repurposed. See [model setup](live_detection/docs/MODEL_SETUP.md).

See [validation](live_detection/docs/VALIDATION.md), [API](live_detection/docs/API.md), and [local deployment limits](live_detection/docs/DEPLOYMENT.md). This is a local research application; public deployment needs authentication and network isolation.
