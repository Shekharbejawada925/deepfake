from collections import Counter
from pathlib import Path
import threading
import time
import json
import hashlib
import numpy as np
from .contract import validate_metadata, default_metadata, LEGACY_ARCHITECTURE
from .vision import FaceExtractor, normalize_crops, same_track


class ModelUnavailable(RuntimeError):
    pass


def verdict(p_fake, meta):
    if p_fake <= meta["real_threshold"]:
        return "LIKELY REAL"
    if p_fake >= meta["fake_threshold"]:
        return "LIKELY FAKE"
    return "INCONCLUSIVE"


class Detector:
    def __init__(self, model_path, device="auto"):
        self.meta = default_metadata()
        self.ready = False
        self.error = "Trained checkpoint missing. See the Model setup guide."
        self.lock = threading.Lock()
        self.model = None
        self.compatibility_mode = False
        self.warning = None
        self.device = "unavailable"
        if not Path(model_path).is_file():
            return
        try:
            import torch
            from .model import VideoModel, LegacyVideoModel, select_device
            torch.set_num_threads(min(4, torch.get_num_threads()))
            payload = torch.load(model_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("Checkpoint must contain a state dictionary.")
            if "metadata" in payload and "state_dict" in payload:
                self.meta = validate_metadata(payload["metadata"])
                if self.meta["architecture"] == LEGACY_ARCHITECTURE:
                    raise ValueError("Load the legacy checkpoint through its matching JSON sidecar.")
                if not self.meta["training_complete"]:
                    raise ValueError("Checkpoint is an initialization, not a trained detector.")
                state = payload["state_dict"]
            else:
                profile = Path(model_path).with_suffix(".json")
                if not profile.is_file():
                    raise ValueError("Legacy checkpoint has no explicit compatibility profile. See docs/MODEL_SETUP.md.")
                self.meta = validate_metadata(json.loads(profile.read_text(encoding="utf-8")))
                with Path(model_path).open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if self.meta["architecture"] != LEGACY_ARCHITECTURE or digest != self.meta["source_sha256"]:
                    raise ValueError("Legacy profile does not match this checkpoint's bytes.")
                state = payload
                self.compatibility_mode = True
                self.warning = ("Legacy checkpoint active. Predictions use the original app's unverified class mapping. "
                                "The original graph scores the final frame only; accuracy has not been measured.")
            self.device = select_device(device)
            self.model = LegacyVideoModel() if self.compatibility_mode else VideoModel(pretrained=False)
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device).eval()
            # Check the selected face detector before enabling inference.
            FaceExtractor(self.meta)
            self.ready = True
            self.error = None
        except Exception as exc:
            self.model = None
            self.error = "Cannot load trained detector: " + str(exc)

    def status(self):
        return {"ready": self.ready, "message": self.error,
                "compatibility_mode": self.compatibility_mode, "warning": self.warning,
                "architecture": self.meta["architecture"],
                "label_mapping_verified": self.meta.get("label_mapping_verified"),
                "color_space": self.meta["color_space"],
                "temporal_behavior": "last_frame_only" if self.compatibility_mode else "chronological_lstm",
                "device": str(self.device), "sequence_length": self.meta["sequence_length"],
                "sample_fps": self.meta["sample_fps"], "calibrated": self.meta["calibrated"],
                "classes": self.meta["classes"], "detector": self.meta["detector"]}

    def require_ready(self):
        if not self.ready:
            raise ModelUnavailable(self.error)

    def score(self, crops):
        self.require_ready()
        import torch
        x = torch.from_numpy(normalize_crops(crops, self.meta)).unsqueeze(0)
        start = time.perf_counter()
        with self.lock, torch.inference_mode():
            x = x.to(self.device)
            if self.compatibility_mode:
                # Exactly equivalent to the original batch-one eval graph;
                # avoid encoding 59 frames that cannot influence its score.
                x = x[:, -1:]
            # Bound encoder memory even for 100-frame windows.
            features = torch.cat([self.model.encode(chunk) for chunk in x[0].split(16)], dim=0)
            logits = self.model.classify(features.unsqueeze(0))
            probabilities = torch.softmax(logits / self.meta["temperature"], dim=1)[0]
            p_fake = float(probabilities[self.meta["classes"].index("fake")].item())
        if not np.isfinite(p_fake):
            raise ValueError("Model returned a non-finite probability.")
        return p_fake, (time.perf_counter()-start)*1000


def prepare_window(samples, extractor, meta):
    reasons = Counter()
    crops, boxes = [], []
    previous = None
    for _, frame in samples:
        item = extractor.extract(frame)
        if item.reason:
            reasons[item.reason] += 1
            previous = None
        else:
            if previous is not None and not same_track(previous, item.box):
                reasons["scene_or_track_change"] += 1
            crops.append(item.crop)
            previous = item.box
        boxes.append(item.box)
    times = np.array([x[0] for x in samples], dtype=float)
    if len(times) > 1:
        delta = np.diff(times)
        interval = 1. / meta["sample_fps"]
        if not np.all(np.isfinite(times)) or np.any(delta <= 0) or np.any(delta > 2.5*interval):
            reasons["irregular_sampling"] += 1
        elif not 0.65 <= (1 / float(np.mean(delta))) / meta["sample_fps"] <= 1.5:
            reasons["irregular_sampling"] += 1
    legacy = meta["architecture"] == LEGACY_ARCHITECTURE
    if (not samples or len(samples) > meta["sequence_length"] or
            (not legacy and len(samples) != meta["sequence_length"])):
        reasons["insufficient_frames"] += 1
    return crops, boxes, dict(reasons)


def analyze_window(samples, detector, extractor):
    detector.require_ready()
    started = time.perf_counter()
    crops, boxes, reasons = prepare_window(samples, extractor, detector.meta)
    result = {"start_seconds": round(samples[0][0],3) if samples else 0,
              "end_seconds": round(samples[-1][0],3) if samples else 0,
              "frames": len(samples), "usable_faces": len(crops),
              "reasons": reasons, "face_box": boxes[-1] if boxes else None,
              "label": "INCONCLUSIVE", "p_fake": None, "inference_ms": None}
    result["compatibility_mode"] = getattr(detector, "compatibility_mode", False)
    result["label_mapping_verified"] = detector.meta.get("label_mapping_verified")
    result["score_scope"] = "final sampled face only" if result["compatibility_mode"] else "temporal face window"
    if not reasons:
        p, ms = detector.score(crops)
        result.update(label=verdict(p, detector.meta), p_fake=round(p,6), inference_ms=round(ms,1))
    result["processing_ms"] = round((time.perf_counter()-started)*1000,1)
    return result


def summarize(windows):
    scored = [x for x in windows if x["p_fake"] is not None]
    labels = {x["label"] for x in windows}
    # Never call a whole clip real if any window was unscorable or ambiguous.
    label = "INCONCLUSIVE"
    if "LIKELY FAKE" in labels:
        label = "FLAGGED SEGMENTS"
    elif windows and labels == {"LIKELY REAL"}:
        label = "LIKELY REAL"
    latencies = [x["processing_ms"] for x in windows]
    return {"label": label, "windows": len(windows), "scored_windows": len(scored),
            "coverage": len(scored) / len(windows) if windows else 0,
            "mean_p_fake": float(np.mean([x["p_fake"] for x in scored])) if scored else None,
            "p95_processing_ms": float(np.percentile(latencies,95)) if latencies else None,
            "note": ("Legacy scores describe the final sampled face in each reporting window. "
                     "Class mapping is assumed from the original app and scores are uncalibrated. "
                     "They do not verify audio or source authenticity.") if any(x.get("compatibility_mode") for x in windows) else
                    "Scores describe sampled facial video windows. They do not verify audio or the entire source."}
