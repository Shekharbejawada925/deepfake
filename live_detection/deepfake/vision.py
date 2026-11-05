"""Shared preprocessing for training, evaluation, uploads and live windows."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FaceFrame:
    crop: object = None
    box: object = None
    reason: str = ""


class FaceExtractor:
    def __init__(self, meta):
        self.meta = meta
        if meta["detector"] == "hog":
            import face_recognition
            self.hog = face_recognition
        else:
            self.hog = None
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self.cascade.empty():
                raise RuntimeError("OpenCV face detector is unavailable.")

    def extract(self, bgr):
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 640.0 / max(height, width))
        small_rgb = cv2.resize(rgb, None, fx=scale, fy=scale) if scale < 1 else rgb
        small_gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
        if self.hog:
            detect_image = small_rgb[:, :, ::-1].copy() if self.meta["color_space"] == "BGR" else small_rgb
            boxes = [(l, t, r-l, b-t) for t, r, b, l in self.hog.face_locations(detect_image, model="hog")]
        else:
            boxes = self.cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(boxes) != 1:
            return FaceFrame(reason="no_face" if len(boxes) == 0 else "multiple_faces")
        x, y, w, h = [int(v / scale) for v in boxes[0]]
        x, y = max(0, x), max(0, y)
        w, h = min(width-x, w), min(height-y, h)
        box = (x / width, y / height, w / width, h / height)
        if min(w, h) < self.meta["min_face_pixels"]:
            return FaceFrame(box=box, reason="face_too_small")
        face_gray = gray[y:y+h, x:x+w]
        if cv2.Laplacian(face_gray, cv2.CV_64F).var() < self.meta["min_blur"]:
            return FaceFrame(box=box, reason="blurred_face")
        if not 20 <= float(face_gray.mean()) <= 235:
            return FaceFrame(box=box, reason="poor_lighting")
        margin = self.meta["crop_margin"]
        dx, dy = int(w * margin), int(h * margin)
        crop = rgb[max(0,y-dy):min(height,y+h+dy), max(0,x-dx):min(width,x+w+dx)]
        if self.meta["color_space"] == "BGR":
            from PIL import Image
            crop = np.asarray(Image.fromarray(crop[:, :, ::-1].copy()).resize((112,112), Image.Resampling.BILINEAR))
        else:
            crop = cv2.resize(crop, (112,112), interpolation=cv2.INTER_AREA)
        return FaceFrame(crop=crop, box=box)


def normalize_crops(crops, meta):
    array = np.asarray(crops, dtype=np.float32) / 255.0
    array = (array - np.asarray(meta["mean"], dtype=np.float32)) / np.asarray(meta["std"], dtype=np.float32)
    return np.ascontiguousarray(array.transpose(0,3,1,2))


def same_track(a, b):
    """Geometric continuity only; this is not identity recognition."""
    if a is None or b is None:
        return False
    x = max(0., min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0]))
    y = max(0., min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    intersection = x*y
    union = a[2]*a[3]+b[2]*b[3]-intersection
    return union > 0 and intersection / union >= 0.1


def resize_frame(frame):
    scale = min(1., 960. / max(frame.shape[:2]))
    return cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame


def iter_video(path, sample_fps, max_seconds=300, stop=None):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video could not be opened. Try an MP4 with H.264 or a readable AVI.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate metadata; remux it before analysis.")
        if fps < sample_fps * 0.9:
            raise ValueError("Video frame rate is below the checkpoint sampling rate. Use a suitably trained model.")
        index, next_sample = 0, 0.
        while not (stop and stop.is_set()):
            ok, frame = cap.read()
            if not ok:
                break
            nominal = index / fps
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
            timestamp = pts if np.isfinite(pts) and pts > 0 else nominal
            index += 1
            if timestamp >= max_seconds:
                break
            if timestamp + 1e-6 >= next_sample:
                yield timestamp, resize_frame(frame)
                # Stay on a fixed time grid. Adding an interval to the decoded
                # timestamp undersamples 25-fps video at 8.33 instead of 10 fps.
                next_sample = (int(timestamp * sample_fps + 1e-6) + 1) / sample_fps
        if index == 0:
            raise ValueError("Video contains no decodable frames.")
    finally:
        cap.release()
