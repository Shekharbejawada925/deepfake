import pytest
import numpy as np
from deepfake.config import Settings
from deepfake.contract import default_metadata
from deepfake.web import create_app
from deepfake.vision import FaceFrame


class StubDetector:
    """Test double only. Never used by app.py or packaged as a model."""
    def __init__(self,p=.05):
        self.meta=default_metadata(10,10); self.ready=True; self.p=p; self.calls=0
    def require_ready(self):
        pass
    def status(self):
        return {"ready":True,"message":None,"sequence_length":10,"sample_fps":10,"device":"test double","calibrated":False,"classes":["fake","real"],"detector":"haar"}
    def score(self,crops):
        self.calls += 1
        return self.p,1.0


class StubExtractor:
    def __init__(self,meta):
        pass
    def extract(self,frame):
        return FaceFrame(crop=np.zeros((112,112,3),dtype=np.uint8),box=(.2,.2,.5,.5))


@pytest.fixture
def app(tmp_path):
    settings=Settings(data_dir=tmp_path/"runtime",model_path=tmp_path/"none.pt",max_jobs=1,stream_hosts=("camera.example",))
    app=create_app(settings,StubDetector(),StubExtractor)
    app.config["TESTING"]=True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth(client):
    return {"X-CSRF-Token":client.get("/api/status").json["csrf"]}
