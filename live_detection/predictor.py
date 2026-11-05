"""Adapter interface to an EXISTING trained model, not a dummy prediction."""
import importlib
import math
import os

class ModelUnavailable(RuntimeError):
    pass

class Predictor:
    def __init__(self):
        self.fn = None
        name = os.getenv("PREDICTOR_MODULE", "")
        if not name:
            self.message = "No trained model connected. Set PREDICTOR_MODULE."
            return
        try:
            module = importlib.import_module(name)
            fn = getattr(module, "predict_frames")
            if hasattr(module, "is_ready") and not module.is_ready():
                raise ModelUnavailable("Trained checkpoint not found; set DEEPFAKE_CHECKPOINT.")
            if not callable(fn):
                raise TypeError("predict_frames must be callable")
            self.fn = fn
            self.message = "Model adapter loaded"
        except (ImportError, AttributeError, TypeError, ModelUnavailable) as exc:
            self.message = f"Adapter unavailable: {exc}"

    @property
    def ready(self):
        return self.fn is not None

    def predict(self, frames):
        if not self.ready:
            raise ModelUnavailable(self.message)
        try:
            score = float(self.fn(frames))
        except ValueError as exc:
            return {"label":"Inconclusive", "fake_probability":None,
                    "frames_analyzed":len(frames), "reason":str(exc),
                    "notice":"No authenticity verdict was produced."}
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Model must return P(fake) between 0 and 1")
        result = "Likely fake" if score >= .65 else "Likely real" if score <= .35 else "Uncertain"
        return {"label": result, "fake_probability": round(score, 4),
                "frames_analyzed": len(frames),
                "notice": "Experimental estimate; not proof of authenticity."}
