from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_path: Path = ROOT / "models/legacy_detector.pt"
    data_dir: Path = ROOT / "data/runtime"
    device: str = "auto"
    max_upload_mb: int = 100
    max_video_seconds: int = 300
    max_stream_seconds: int = 600
    max_jobs: int = 2
    retention_seconds: int = 3600
    stream_hosts: tuple = ()

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", str(ROOT / "models/legacy_detector.pt"))).resolve(),
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data/runtime"))).resolve(),
            device=os.getenv("DEVICE", "auto"),
            max_upload_mb=max(1, int(os.getenv("MAX_UPLOAD_MB", "100"))),
            max_video_seconds=max(1, int(os.getenv("MAX_VIDEO_SECONDS", "300"))),
            max_stream_seconds=max(1, int(os.getenv("MAX_STREAM_SECONDS", "600"))),
            max_jobs=max(1, int(os.getenv("MAX_JOBS", "2"))),
            retention_seconds=max(60, int(os.getenv("RETENTION_SECONDS", "3600"))),
            stream_hosts=tuple(x.strip().lower() for x in os.getenv("STREAM_ALLOWED_HOSTS", "").split(",") if x.strip()),
        )
