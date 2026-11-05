"""Upload a local video for the same model inference used by the live camera."""
import os
from pathlib import Path
import tempfile
import time

from flask import Blueprint, current_app, jsonify, request
from PIL import Image

uploads = Blueprint("video_upload", __name__)
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FRAMES_TO_SCAN = 900

@uploads.post("/api/video/predict")
def upload_predict():
    from app import predictor
    if not predictor.ready:
        return jsonify(error=predictor.message), 503
    f=request.files.get("file")
    if f is None or not f.filename:
        return jsonify(error="Select a video file"), 400
    suffix=Path(f.filename).suffix.lower()
    if suffix not in ALLOWED:
        return jsonify(error="Unsupported video format"), 400
    try:
        import cv2
    except ImportError:
        return jsonify(error="Install requirements-model.txt (OpenCV required)"), 503
    name=None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as handle:
            name=handle.name
            # Request limits are enforced by Flask on this endpoint.
            while True:
                block=f.stream.read(1024*1024)
                if not block: break
                handle.write(block)
        cap=cv2.VideoCapture(name)
        if not cap.isOpened():
            return jsonify(error="Unable to decode the uploaded video"), 400
        try:
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames=[]
            start=time.monotonic()
            # Sample 8 roughly spaced frames, bounded to prevent excessively long tasks.
            if 8 <= count <= 100000:
                indices=[round(i*(count-1)/7) for i in range(8)]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
                    ok,frame=cap.read()
                    if ok:
                        frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                    if time.monotonic()-start>25: break
            else:
                sampled=[]
                for i in range(MAX_FRAMES_TO_SCAN):
                    ok,frame=cap.read()
                    if not ok or time.monotonic()-start>25: break
                    if i%20==0:
                        sampled.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
                if len(sampled)>=8:
                    positions=[round(i*(len(sampled)-1)/7) for i in range(8)]
                    frames=[sampled[idx] for idx in positions]
            if len(frames)!=8:
                return jsonify(error="Insufficient decodable frames or decoder timed out"),400
            result=predictor.predict(frames)
            result["source"]="uploaded_video"
            return jsonify(result)
        finally:
            cap.release()
    except Exception:
        current_app.logger.exception("Video decoding/inference failed")
        return jsonify(error="Video processing failed"),500
    finally:
        if name:
            try: os.remove(name)
            except OSError: pass
