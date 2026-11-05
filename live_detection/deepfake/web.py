import base64
import binascii
import io
import json
import os
import secrets
import threading
import numpy as np
import cv2
from flask import Flask, jsonify, render_template, request, session, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename
from .config import Settings, ROOT
from .inference import Detector, ModelUnavailable, analyze_window
from .vision import FaceExtractor
from .streams import validate_stream_url
from .jobs import JobManager, CapacityError


def create_app(settings=None, detector=None, extractor_factory=FaceExtractor):
    settings = settings or Settings.from_env()
    app = Flask(__name__,template_folder=str(ROOT/"templates"),static_folder=str(ROOT/"static"))
    app.config.update(SECRET_KEY=os.getenv("SECRET_KEY") or secrets.token_hex(32),
                      MAX_CONTENT_LENGTH=settings.max_upload_mb*1024*1024,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
                      SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE","0") == "1")
    detector = detector if detector is not None else Detector(settings.model_path,settings.device)
    manager = JobManager(detector,settings,extractor_factory)
    camera_slots = threading.BoundedSemaphore(settings.max_jobs)
    app.extensions["detector"], app.extensions["jobs"] = detector, manager

    def identity():
        if "owner" not in session:
            session["owner"], session["csrf"] = secrets.token_hex(24), secrets.token_hex(24)
        return session["owner"]

    @app.before_request
    def guard():
        identity()
        if request.method in ("POST","PUT","DELETE","PATCH"):
            token = request.headers.get("X-CSRF-Token", "")
            if not secrets.compare_digest(token, session["csrf"]):
                return jsonify(error="Session token missing or expired. Reload the page."),403

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    @app.get("/live")
    def index():
        return render_template("index.html")

    @app.get("/guide")
    def guide():
        return render_template("guide.html")

    @app.get("/api/status")
    def status():
        return jsonify(model=detector.status(),csrf=session["csrf"],
                       limits={"upload_mb":settings.max_upload_mb,"video_seconds":settings.max_video_seconds,
                               "stream_seconds":settings.max_stream_seconds},
                       streaming_enabled=bool(settings.stream_hosts))

    @app.get("/api/health")
    def health():
        return jsonify(status="ok",model_ready=detector.ready)

    @app.post("/api/uploads")
    def upload():
        detector.require_ready()
        item = request.files.get("file")
        if item is None or not item.filename:
            raise ValueError("Choose a video file.")
        name = secure_filename(item.filename)
        suffix = os.path.splitext(name)[1].lower()
        if suffix not in (".mp4",".mov",".avi",".mkv",".webm"):
            raise ValueError("Choose an MP4, MOV, AVI, MKV or WebM video.")
        job = manager.reserve(identity(),"upload",name)
        source = job.folder / ("source"+suffix)
        try:
            item.save(source)
            if source.stat().st_size == 0:
                raise ValueError("Uploaded file is empty.")
            manager.start(job,source)
        except Exception:
            manager.abort_reserved(job)
            raise
        return jsonify(id=job.id),202

    @app.post("/api/streams")
    def stream():
        detector.require_ready()
        payload = request.get_json(silent=True)
        if not isinstance(payload,dict):
            raise ValueError("Send a JSON object with a url.")
        url = validate_stream_url(payload.get("url"),settings.stream_hosts)
        job = manager.reserve(identity(),"stream","Direct live stream")
        try:
            manager.start(job,url)
        except Exception:
            manager.abort_reserved(job)
            raise
        return jsonify(id=job.id),202

    @app.get("/api/jobs/<key>")
    def get_job(key):
        return jsonify(manager.get(key,identity()).snapshot())

    @app.post("/api/jobs/<key>/stop")
    def stop_job(key):
        job = manager.get(key,identity())
        with job.lock:
            if job.state in ("queued","running"):
                job.message = "Stop requested"
                job.cancel_requested = True
                job.stop.set()
        return jsonify(status="stopping")

    @app.delete("/api/jobs/<key>")
    def delete_job(key):
        manager.delete(manager.get(key,identity()))
        return jsonify(status="deleted")

    @app.get("/api/jobs/<key>/preview.jpg")
    def preview(key):
        path = manager.get(key,identity()).folder / "preview.jpg"
        if not path.is_file():
            raise KeyError(key)
        return send_file(path.resolve(),mimetype="image/jpeg")

    @app.get("/api/jobs/<key>/report")
    def report(key):
        data = manager.get(key,identity()).snapshot()
        data["model"] = detector.status()
        payload = io.BytesIO(json.dumps(data,indent=2).encode("utf-8"))
        return send_file(payload,mimetype="application/json",as_attachment=True,download_name="analysis-"+key[:8]+".json")

    @app.post("/api/camera/window")
    @app.post("/api/live/window")
    def camera():
        detector.require_ready()
        if request.content_length and request.content_length > 8*1024*1024:
            raise ValueError("Camera window is too large. Reduce capture resolution.")
        payload = request.get_json(silent=True)
        if not isinstance(payload,dict):
            raise ValueError("Send a camera window as JSON.")
        frames, times = payload.get("frames"), payload.get("timestamps")
        n = detector.meta["sequence_length"]
        if not isinstance(frames,list) or len(frames) != n or not isinstance(times,list) or len(times) != n:
            raise ValueError(f"A camera window requires exactly {n} frames and timestamps.")
        if not camera_slots.acquire(blocking=False):
            raise CapacityError("Camera processing is busy. Try again shortly.")
        try:
            samples = []
            for encoded, stamp in zip(frames,times):
                if type(stamp) not in (float,int) or not np.isfinite(stamp) or stamp < 0:
                    raise ValueError("Camera timestamps must be finite seconds.")
                if not isinstance(encoded,str) or len(encoded) > 250000:
                    raise ValueError("Invalid or oversized camera frame.")
                try:
                    raw = base64.b64decode(encoded,validate=True)
                except (binascii.Error,ValueError) as exc:
                    raise ValueError("Invalid camera image encoding.") from exc
                # Read dimensions before decompression to reject oversized images.
                from PIL import Image
                try:
                    with Image.open(io.BytesIO(raw)) as probe:
                        if probe.format != "JPEG" or probe.width > 960 or probe.height > 960:
                            raise ValueError("Camera JPEG must be at most 960 pixels per side.")
                except (OSError,Image.DecompressionBombError) as exc:
                    raise ValueError("Unreadable camera JPEG.") from exc
                bgr = cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Unreadable camera frame.")
                samples.append((float(stamp),bgr))
            result = analyze_window(samples,detector,extractor_factory(detector.meta))
            return jsonify(result)
        finally:
            camera_slots.release()

    @app.errorhandler(ModelUnavailable)
    def missing_model(exc):
        return jsonify(error=str(exc),code="model_unavailable"),503

    @app.errorhandler(CapacityError)
    def capacity(exc):
        return jsonify(error=str(exc)),429

    @app.errorhandler(ValueError)
    def bad_input(exc):
        return jsonify(error=str(exc)),400

    @app.errorhandler(KeyError)
    def missing(exc):
        return jsonify(error="Session not found or no longer available."),404

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(exc):
        return jsonify(error=f"Request exceeds the {settings.max_upload_mb} MB limit."),413

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error=exc.description),exc.code

    @app.errorhandler(Exception)
    def unexpected(exc):
        app.logger.exception("Unexpected application error")
        return jsonify(error="Processing failed. Check the server log for details."),500

    return app
