"""A checkpoint records the preprocessing and semantics it was trained with."""
import math

ARCHITECTURE = "resnext50_lstm_v2"
LEGACY_ARCHITECTURE = "resnext50_lstm_legacy_compat"
LENGTHS = (10, 20, 40, 60, 80, 100)


def default_metadata(sequence_length=60, sample_fps=10.0, detector="haar"):
    return {
        "format_version": 2,
        "architecture": ARCHITECTURE,
        "classes": ["fake", "real"],
        "sequence_length": sequence_length,
        "sample_fps": sample_fps,
        "image_size": 112,
        "color_space": "RGB",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "hidden_size": 2048,
        "lstm_layers": 1,
        "dropout": 0.4,
        "detector": detector,
        "crop_margin": 0.15,
        "min_face_pixels": 64,
        "min_blur": 20.0,
        "real_threshold": 0.2,
        "fake_threshold": 0.8,
        "temperature": 1.0,
        "calibrated": False,
        "training_complete": False,
    }


def legacy_metadata(source_sha256):
    """Explicit source-code assumptions, not recovered training metadata."""
    # The legacy graph has no temporal dependence at batch size one. Ten-frame
    # reporting windows give a one-second update without changing its weights.
    meta = default_metadata(10, 10.0, "haar")
    meta.update(architecture=LEGACY_ARCHITECTURE, color_space="BGR",
                crop_margin=0.0, resize_method="PIL_BILINEAR",
                training_complete=False, source_sha256=source_sha256,
                profile_name="Supplied legacy checkpoint",
                temporal_behavior="last_frame_only",
                label_mapping_verified=False,
                label_mapping_source="Uploaded app.py: class 0=fake, class 1=real; training labels unavailable",
                preprocessing_source="Uploaded app.py BGR crops and PIL bilinear resize; portable Haar detection replaces HOG",
                filename_sequence_length=60,
                sampling_source="10 fps reporting profile; original training rate unknown",
                window_policy="10 sampled frames per live update; score final frame only; partial uploads allowed")
    return meta


def validate_metadata(meta):
    if not isinstance(meta, dict):
        raise ValueError("Missing checkpoint metadata. Retrain with tools.train.")
    required = set(default_metadata())
    if required - meta.keys():
        raise ValueError("Checkpoint metadata is incomplete: " + ", ".join(sorted(required - meta.keys())))
    legacy = meta["architecture"] == LEGACY_ARCHITECTURE
    if meta["architecture"] not in (ARCHITECTURE, LEGACY_ARCHITECTURE) or meta["format_version"] != 2:
        raise ValueError("Unsupported checkpoint architecture or format; see docs/MODEL_SETUP.md.")
    if meta["classes"] not in (["fake", "real"], ["real", "fake"]):
        raise ValueError("Class order must explicitly name fake and real.")
    if meta["sequence_length"] not in LENGTHS or type(meta["sequence_length"]) is not int:
        raise ValueError("Unsupported sequence length.")
    if meta["color_space"] != ("BGR" if legacy else "RGB") or meta["image_size"] != 112:
        raise ValueError("Color space or crop size does not match the model profile.")
    if legacy:
        digest = meta.get("source_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Legacy profile requires the exact source checkpoint SHA-256.")
        if meta.get("temporal_behavior") != "last_frame_only" or meta.get("resize_method") != "PIL_BILINEAR":
            raise ValueError("Legacy execution assumptions must be explicit.")
        if meta.get("label_mapping_verified") is not False:
            raise ValueError("This legacy profile's class mapping has not been verified on labeled data.")
        if meta["crop_margin"] != 0.0 or meta["training_complete"] or meta["calibrated"]:
            raise ValueError("Legacy profiles cannot be represented as retrained or calibrated checkpoints.")
    if meta["hidden_size"] != 2048 or meta["lstm_layers"] != 1 or meta["dropout"] != 0.4:
        raise ValueError("Architecture metadata does not match the paper implementation.")
    for key in ("sample_fps", "crop_margin", "min_face_pixels", "min_blur", "temperature", "real_threshold", "fake_threshold"):
        if not isinstance(meta[key], (float, int)) or isinstance(meta[key], bool) or not math.isfinite(meta[key]):
            raise ValueError("Invalid numeric metadata: " + key)
    if not 1 <= meta["sample_fps"] <= 30 or not 0 < meta["temperature"] <= 10:
        raise ValueError("Invalid sample rate or temperature.")
    if not 0 <= meta["real_threshold"] < 0.5 < meta["fake_threshold"] <= 1:
        raise ValueError("Decision thresholds must leave an uncertainty interval.")
    if not 0 <= meta["crop_margin"] <= 0.5 or not 16 <= meta["min_face_pixels"] <= 512 or meta["min_blur"] < 0:
        raise ValueError("Invalid face quality configuration.")
    if meta["detector"] not in ("haar", "hog"):
        raise ValueError("Detector must be haar or hog.")
    if meta["mean"] != [0.485, 0.456, 0.406] or meta["std"] != [0.229, 0.224, 0.225]:
        raise ValueError("Normalization does not match the supported model.")
    if type(meta["calibrated"]) is not bool or type(meta["training_complete"]) is not bool:
        raise ValueError("Training and calibration flags must be booleans.")
    return dict(meta)
