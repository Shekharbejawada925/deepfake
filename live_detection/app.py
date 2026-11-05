import base64
import io
import logging
from flask import Flask, jsonify, render_template, request
from PIL import Image
from predictor import Predictor, ModelUnavailable

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
from video_upload import uploads
app.register_blueprint(uploads)
predictor = Predictor()

@app.get("/live")
def live():
    return render_template("live.html")

@app.get("/api/live/status")
def status():
    return jsonify(ready=predictor.ready, message=predictor.message)

@app.post("/api/live/predict")
def predict():
    payload = request.get_json(silent=True)
    sources = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(sources, list) or len(sources) != 8:
        return jsonify(error="Expected exactly 8 JPEG frames"), 400
    frames = []
    try:
        for src in sources:
            if not isinstance(src, str) or not src.startswith("data:image/jpeg;base64,"):
                raise ValueError("Only JPEG data URLs accepted")
            raw = base64.b64decode(src.split(",", 1)[1], validate=True)
            if len(raw) > 800_000:
                raise ValueError("Frame exceeds 800KB")
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                if max(im.size) > 1280:
                    raise ValueError("Image dimensions exceed limit")
                frames.append(im.convert("RGB").copy())
    except (ValueError, OSError) as exc:
        return jsonify(error="Invalid input: " + str(exc)), 400
    try:
        return jsonify(predictor.predict(frames))
    except ModelUnavailable as exc:
        return jsonify(error=str(exc)), 503
    except Exception:
        logging.exception("Inference error")
        return jsonify(error="Inference failed; inspect logs"), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
