"""Direct media streams with bounded queues, open/read timeouts and cancellation."""
from collections import deque
from urllib.parse import urlsplit
import threading
import time
import cv2
from .vision import resize_frame


def validate_stream_url(url, allowed_hosts):
    if not isinstance(url, str) or not url or len(url) > 2048 or any(ord(c) < 33 for c in url):
        raise ValueError("Enter a valid direct stream URL.")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("rtsp", "http", "https") or not parsed.hostname or parsed.fragment:
        raise ValueError("Use a direct RTSP, HTTP or HTTPS media stream.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("Invalid stream port.") from exc
    if parsed.hostname.lower() not in allowed_hosts:
        raise ValueError("This stream host is not enabled. Add its exact hostname to STREAM_ALLOWED_HOSTS and restart.")
    return url


def open_stream(url):
    # Timeout properties must be supplied at open time to a supporting backend.
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot open the stream. Check the media URL, credentials, codec and connectivity.")
    return cap


class StreamReader:
    def __init__(self, url, fps, capacity, stop, opener=open_stream):
        self.url, self.fps, self.stop, self.opener = url, fps, stop, opener
        self.queue = deque(maxlen=capacity)
        self.condition = threading.Condition()
        self.done, self.error, self.dropped = False, None, 0
        self.thread = threading.Thread(target=self._read, daemon=True, name="media-reader")

    def _read(self):
        cap = None
        try:
            cap = self.opener(self.url)
            started, next_sample = time.monotonic(), 0.
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "Stream ended or read timed out. Restart the session to reconnect."
                    break
                timestamp = time.monotonic()-started
                if timestamp + 1e-6 < next_sample:
                    continue
                sample = (timestamp, resize_frame(frame))
                next_sample = timestamp+1/self.fps
                with self.condition:
                    if len(self.queue) == self.queue.maxlen:
                        self.dropped += 1
                    self.queue.append(sample)
                    self.condition.notify_all()
        except Exception:
            # Do not return URLs or embedded credentials through the public API.
            self.error = "Stream could not be read. Verify its address, credentials and supported codec."
        finally:
            if cap is not None:
                cap.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def samples(self, max_seconds):
        self.thread.start()
        started = time.monotonic()
        try:
            while not self.stop.is_set() and time.monotonic()-started < max_seconds:
                with self.condition:
                    self.condition.wait_for(lambda: self.queue or self.done or self.stop.is_set(), timeout=0.25)
                    sample = self.queue.popleft() if self.queue else None
                    done = self.done
                if sample is not None:
                    yield sample
                elif done:
                    break
        finally:
            self.stop.set()
            self.thread.join(timeout=5.5)
