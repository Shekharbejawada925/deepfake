"""Adapter for the original ResNeXt50/LSTM checkpoint.

The original LSTM has batch_first=False and bias=False due to the positional
fourth argument. This is a compatibility adapter, not a corrected temporal model.
A faithful original checkpoint load is required before predictions are enabled.
"""
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image

_MODEL = None
_LOCK = threading.Lock()

def _build_and_load():
    import torch
    from torch import nn
    from torchvision import models
    class LegacyModel(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = models.resnext50_32x4d(weights=None)
            self.model = nn.Sequential(*list(backbone.children())[:-2])
            # Original code: nn.LSTM(2048, 2048, 1, False).
            self.lstm = nn.LSTM(2048, 2048, 1, bias=False, batch_first=False)
            self.relu = nn.LeakyReLU()
            self.dp = nn.Dropout(0.4)
            self.linear1 = nn.Linear(2048, 2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            b, t, c, h, w = x.shape
            fmap = self.model(x.reshape(b*t, c, h, w))
            z = self.avgpool(fmap).reshape(b, t, 2048)
            temporal, _ = self.lstm(z)
            return self.dp(self.linear1(temporal[:, -1, :]))

    checkpoint = Path(os.getenv("DEEPFAKE_CHECKPOINT", "models/df_model.pt")).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Set DEEPFAKE_CHECKPOINT.")
    model = LegacyModel()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected a state_dict checkpoint, not a full model object")
    model.load_state_dict(state, strict=True)
    model.eval()
    device = torch.device("mps" if os.getenv("DEEPFAKE_USE_MPS") == "1" and torch.backends.mps.is_available() else "cpu")
    return model.to(device), device

def _get_model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            _MODEL = _build_and_load()
        return _MODEL

def _face_crop(image):
    """Haar face detection used as an optional dependency-free fallback.

    Input is RGB Pillow image. Fail closed if no or multiple faces.
    """
    import cv2
    rgb = np.asarray(image.convert("RGB"))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes = detector.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(35,35))
    if len(boxes) != 1:
        raise ValueError(f"Expected exactly one detectable face; found {len(boxes)}")
    x,y,w,h = [int(v) for v in boxes[0]]
    # 10% margin to preserve face boundary context.
    margin = int(max(w,h)*0.1)
    x0,y0=max(0,x-margin),max(0,y-margin)
    x1,y1=min(rgb.shape[1],x+w+margin),min(rgb.shape[0],y+h+margin)
    crop=rgb[y0:y1,x0:x1]
    if crop.size == 0:
        raise ValueError("Empty face crop")
    if cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() < 20:
        raise ValueError("Face too blurry for a reliable score")
    return Image.fromarray(crop).resize((112,112))

def predict_frames(frames):
    """Return P(fake), assuming original app's class mapping 0=fake, 1=real.

    Mapping is NOT verified against training labels. Do not use as a reliable
    authenticity verdict until checked on labelled holdout videos.
    """
    import torch
    from torchvision.transforms import functional as F
    model,device=_get_model()
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]
    faces=[F.normalize(F.to_tensor(_face_crop(im)),mean,std) for im in frames]
    batch=torch.stack(faces).unsqueeze(0).to(device)
    with _LOCK, torch.inference_mode():
        logits=model(batch)
        probabilities=torch.softmax(logits,dim=1)[0]
        return float(probabilities[0].item())
