# Live video detection module

Adds an experimental **webcam capture and inference UI** to the Deepfake project without overwriting your existing Flask application or uploaded-video workflow.

## Launch
```bash
cd live_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PREDICTOR_MODULE=my_model_adapter python app.py
```
Visit http://127.0.0.1:5000/live. Webcam requires browser permission and localhost or HTTPS.

## Connect actual trained CNN–LSTM

Place a `my_model_adapter.py` file in this folder defining `predict_frames(frames)` which receives **8 Pillow RGB images** and returns the probability of **FAKE** as a float from 0 to 1. It must load the existing `model/df_model.pt` weights using the **exact same architecture, cropping, normalization, sequence length and class mapping** used in training. If the model expects 20/32 frames, change both the JavaScript collection batch and Flask validation before enabling inference. The adapter should cache the model, call `eval()` and use `torch.inference_mode()`. Do not use an untrained model or fixed prediction.

## Integration with existing Flask app

Merge the `/live`, `/api/live/status` and `/api/live/predict` routes into your existing application or register a blueprint, and copy the HTML template to your templates folder. Add a link to `/live` in the existing UI. Do not run both Flask apps on the same port.

**Current limitations:** This repo does not contain the original trained model or full prediction code. Model results are deliberately disabled until the adapter is implemented. There is no RTSP or third-party stream support in this branch. This application is intended for local development, not public deployment; add authentication, CSRF protections, request limits, and HTTPS before deploying. Camera frames are not persisted by this module.
