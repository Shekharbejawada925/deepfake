from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
import shutil
import threading
import time
import uuid
import cv2
from .inference import analyze_window, summarize
from .vision import FaceExtractor, iter_video
from .streams import StreamReader


class CapacityError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    owner: str
    kind: str
    name: str
    folder: Path
    state: str = "queued"
    message: str = "Waiting for processing"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    windows: list = field(default_factory=list)
    dropped: int = 0
    processed_seconds: float = 0.
    cancel_requested: bool = False
    stop: object = field(default_factory=threading.Event)
    lock: object = field(default_factory=threading.RLock)

    def snapshot(self):
        with self.lock:
            return {"id": self.id, "kind": self.kind, "name": self.name,
                    "state": self.state, "message": self.message,
                    "created": self.created, "updated": self.updated,
                    "dropped_samples": self.dropped,
                    "processed_seconds": round(self.processed_seconds,2),
                    "timeline": copy.deepcopy(self.windows),
                    "summary": summarize(self.windows)}


class JobManager:
    def __init__(self, detector, settings, extractor_factory=FaceExtractor):
        self.detector, self.settings = detector, settings
        self.extractor_factory = extractor_factory
        self.jobs, self.lock = {}, threading.RLock()
        self.slots = threading.BoundedSemaphore(settings.max_jobs)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        # Runtime files left by an earlier server process cannot be addressed by a new session.
        for folder in settings.data_dir.glob("job-*"):
            if folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)

    def cleanup(self):
        with self.lock:
            old = [k for k,j in self.jobs.items() if j.state in ("complete","stopped","error")
                   and time.time()-j.updated > self.settings.retention_seconds]
            for key in old:
                shutil.rmtree(self.jobs.pop(key).folder, ignore_errors=True)

    def reserve(self, owner, kind, name):
        self.cleanup()
        if not self.slots.acquire(blocking=False):
            raise CapacityError("All processing slots are busy. Stop a session or try again shortly.")
        try:
            with self.lock:
                if len(self.jobs) >= 100:
                    raise CapacityError("Session history is full. Delete a finished session before continuing.")
                key = uuid.uuid4().hex
                folder = self.settings.data_dir / ("job-"+key)
                folder.mkdir()
                job = Job(key,owner,kind,name,folder)
                self.jobs[key] = job
                return job
        except Exception:
            self.slots.release()
            raise

    def abort_reserved(self, job):
        with self.lock:
            self.jobs.pop(job.id, None)
        shutil.rmtree(job.folder, ignore_errors=True)
        self.slots.release()

    def get(self, key, owner):
        self.cleanup()
        with self.lock:
            job = self.jobs.get(key)
            if job is None or job.owner != owner:
                raise KeyError(key)
            return job

    def delete(self, job):
        with job.lock:
            if job.state not in ("complete","error","stopped"):
                raise ValueError("Stop this session and wait for it to finish before deleting it.")
            with self.lock:
                self.jobs.pop(job.id, None)
            shutil.rmtree(job.folder, ignore_errors=True)

    def start(self, job, source):
        threading.Thread(target=self._run, args=(job,source),daemon=True,name="analysis-"+job.id[:8]).start()

    def _preview(self, job, sample, result=None):
        frame = sample[1].copy()
        if result and result.get("face_box"):
            x,y,w,h = result["face_box"]
            height,width = frame.shape[:2]
            color = (60,185,255) if result["label"] == "INCONCLUSIVE" else ((80,200,70) if result["label"] == "LIKELY REAL" else (60,60,235))
            cv2.rectangle(frame,(int(x*width),int(y*height)),(int((x+w)*width),int((y+h)*height)),color,2)
        ok, encoded = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,80])
        if ok:
            temporary = job.folder / "preview.tmp"
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(job.folder / "preview.jpg")

    def _run(self, job, source):
        reader, samples, pending = None, None, []
        user_stopped = False
        end_state, end_message = "error", "Processing did not finish"
        try:
            with job.lock:
                job.state, job.message = "running", "Collecting face windows"
            if job.cancel_requested:
                end_state, end_message = "stopped", "Stopped by user"
                return
            extractor = self.extractor_factory(self.detector.meta)
            n, fps = self.detector.meta["sequence_length"], self.detector.meta["sample_fps"]
            if job.kind == "stream":
                reader = StreamReader(source,fps,n*2,job.stop)
                samples = reader.samples(self.settings.max_stream_seconds)
            else:
                samples = iter_video(source,fps,self.settings.max_video_seconds,job.stop)
            count = 0
            for sample in samples:
                if job.stop.is_set():
                    user_stopped = True
                    break
                pending.append(sample)
                count += 1
                with job.lock:
                    job.processed_seconds, job.updated = sample[0], time.time()
                    job.dropped = reader.dropped if reader else 0
                if len(pending) == 1:
                    self._preview(job,sample)
                if len(pending) == n:
                    result = analyze_window(pending,self.detector,extractor)
                    with job.lock:
                        job.windows.append(result)
                        job.message = "Analyzing sampled video windows"
                    self._preview(job,sample,result)
                    pending = []
            user_stopped = job.cancel_requested
            if count == 0 and not user_stopped:
                raise ValueError(reader.error if reader and reader.error else "No decodable video frames were received.")
            if pending:
                result = analyze_window(pending,self.detector,extractor)
                with job.lock:
                    job.windows.append(result)
            end_state = "stopped" if user_stopped else "complete"
            end_message = ("Stopped by user" if user_stopped else
                           (reader.error if reader and reader.error else "Analysis finished; configured duration limits apply"))
        except Exception as exc:
            end_state, end_message = "error", str(exc)
        finally:
            try:
                if samples is not None and hasattr(samples,"close"):
                    samples.close()
                if job.kind == "upload":
                    Path(source).unlink(missing_ok=True)
                # Publish a terminal state only after capture and source cleanup.
                # Keep deletion serialized with the final report write.
                with job.lock:
                    job.state, job.message = end_state, end_message
                    job.updated = time.time()
                    (job.folder / "report.json").write_text(json.dumps(job.snapshot(),indent=2),encoding="utf-8")
            finally:
                self.slots.release()
