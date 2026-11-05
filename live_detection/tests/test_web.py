import base64
import io
import time
import cv2
import numpy as np
from deepfake.config import Settings
from deepfake.web import create_app


def test_no_model_still_serves_ui_but_never_scores(tmp_path):
    app=create_app(Settings(model_path=tmp_path/"missing.pt",data_dir=tmp_path/"runtime"))
    client=app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/live").status_code == 200
    assert client.get("/guide").status_code == 200
    response=client.get("/api/status").json
    assert response["model"]["ready"] is False
    response=client.post("/api/uploads",headers={"X-CSRF-Token":response["csrf"]})
    assert response.status_code == 503
    assert "p_fake" not in response.json


def test_post_requires_session_token(client):
    assert client.post("/api/streams",json={"url":"rtsp://camera.example/live"}).status_code == 403


def test_file_validation(client,auth):
    assert client.post("/api/uploads",headers=auth).status_code == 400
    assert client.post("/api/uploads",headers=auth,data={"file":(io.BytesIO(b"x"),"test.py")}).status_code == 400
    assert client.post("/api/uploads",headers=auth,data={"file":(io.BytesIO(),"empty.mp4")}).status_code == 400


def test_stream_rejects_unapproved_hosts_and_non_media_schemes(client,auth):
    for url in ("file:///etc/passwd","http://127.0.0.1:8000/private","https://youtube.com/watch?v=example"):
        assert client.post("/api/streams",headers=auth,json={"url":url}).status_code == 400


def test_camera_validates_images(client,auth):
    assert client.post("/api/camera/window",headers=auth,json={"frames":[],"timestamps":[]}).status_code == 400
    assert client.post("/api/camera/window",headers=auth,json={"frames":["broken"]*10,"timestamps":[i/10 for i in range(10)]}).status_code == 400


def test_camera_api_complete_window(client,auth):
    _,jpeg=cv2.imencode(".jpg",np.zeros((64,64,3),np.uint8))
    encoded=base64.b64encode(jpeg).decode()
    response=client.post("/api/camera/window",headers=auth,json={"frames":[encoded]*10,"timestamps":[i/10 for i in range(10)]})
    assert response.status_code == 200
    assert response.json["frames"] == 10 and response.json["label"] == "LIKELY REAL"


def test_live_screen_window_uses_same_pipeline_and_security(client,auth):
    _,jpeg=cv2.imencode(".jpg",np.zeros((64,64,3),np.uint8))
    encoded=base64.b64encode(jpeg).decode()
    body={"frames":[encoded]*10,"timestamps":[i/10 for i in range(10)]}
    assert client.post("/api/live/window",json=body).status_code == 403
    live=client.post("/api/live/window",headers=auth,json=body)
    camera=client.post("/api/camera/window",headers=auth,json=body)
    assert live.status_code == camera.status_code == 200
    for key in ("label","p_fake","frames","score_scope","reasons"):
        assert live.json[key] == camera.json[key]
    body["timestamps"]=[0]*10
    result=client.post("/api/live/window",headers=auth,json=body)
    assert result.json["p_fake"] is None
    assert "irregular_sampling" in result.json["reasons"]


def test_cancel_before_worker_starts_removes_upload(app,client,auth):
    manager=app.extensions["jobs"]
    with client.session_transaction() as session:
        owner=session["owner"]
    job=manager.reserve(owner,"upload","cancel.avi")
    source=job.folder/"source.avi"; source.write_bytes(b"not decoded")
    response=client.post("/api/jobs/"+job.id+"/stop",headers=auth)
    assert response.status_code == 200
    manager._run(job,source)
    assert job.state == "stopped" and not source.exists() and job.windows == []


def test_stream_route_runs_complete_worker_and_stops(app,client,auth,monkeypatch):
    from deepfake import jobs
    class StubReader:
        error=None; dropped=2
        def __init__(self,source,fps,n,stop):
            self.stop=stop
        def samples(self,max_seconds):
            for i in range(10):
                yield i/10,np.zeros((64,64,3),np.uint8)
            self.stop.set()
    monkeypatch.setattr(jobs,"StreamReader",StubReader)
    response=client.post("/api/streams",headers=auth,json={"url":"rtsp://camera.example/live"})
    assert response.status_code == 202
    key=response.json["id"]
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        data=client.get("/api/jobs/"+key).json
        if data["state"] in ("complete","error"): break
        time.sleep(.02)
    assert data["state"] == "complete" and len(data["timeline"]) == 1
    assert data["dropped_samples"] == 2


def test_upload_job_completes_report_is_private_and_source_removed(tmp_path,app,client,auth):
    path=tmp_path/"fixture.avi"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(64,64))
    for i in range(12): writer.write(np.full((64,64,3),i*10,np.uint8))
    writer.release()
    response=client.post("/api/uploads",headers=auth,data={"file":(io.BytesIO(path.read_bytes()),"../../fixture.avi")})
    assert response.status_code == 202
    key=response.json["id"]
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        data=client.get("/api/jobs/"+key).json
        if data["state"] in ("complete","error"): break
        time.sleep(.02)
    assert data["state"] == "complete",data
    assert len(data["timeline"]) == 2
    assert data["summary"]["label"] == "INCONCLUSIVE"  # trailing incomplete window
    job=app.extensions["jobs"].jobs[key]
    assert not list(job.folder.glob("source.*"))
    assert client.get("/api/jobs/"+key+"/report").status_code == 200
    stranger=app.test_client()
    assert stranger.get("/api/jobs/"+key).status_code == 404
    assert stranger.get("/api/jobs/"+key+"/preview.jpg").status_code == 404
    assert client.delete("/api/jobs/"+key,headers=auth).status_code == 200
    assert not job.folder.exists()


def test_preview_uses_workspace_relative_runtime_path(tmp_path, monkeypatch):
    from pathlib import Path
    from conftest import StubDetector
    monkeypatch.chdir(tmp_path)
    app = create_app(Settings(data_dir=Path('relative-runtime')), detector=StubDetector())
    client = app.test_client()
    client.get('/api/status')
    with client.session_transaction() as session:
        owner = session['owner']
    manager = app.extensions['jobs']
    job = manager.reserve(owner, 'upload', 'test.avi')
    (job.folder / 'preview.jpg').write_bytes(b'jpeg-test-bytes')
    try:
        response = client.get('/api/jobs/' + job.id + '/preview.jpg')
        assert response.status_code == 200 and response.data == b'jpeg-test-bytes'
    finally:
        manager.abort_reserved(job)
