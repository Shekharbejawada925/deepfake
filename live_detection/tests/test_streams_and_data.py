import csv
import threading
import time
import numpy as np
import pytest
from deepfake.streams import StreamReader
from tools.dataset import read_manifest


def test_stream_reader_yields_and_releases_capture():
    class Capture:
        count=0; released=False
        def read(self):
            time.sleep(.02); self.count+=1
            return (True,np.zeros((32,32,3),np.uint8)) if self.count <= 8 else (False,None)
        def release(self): self.released=True
    capture=Capture(); stop=threading.Event()
    reader=StreamReader("test-source",30,2,stop,opener=lambda _:capture)
    values=list(reader.samples(2))
    assert len(values)>0 and capture.released and reader.done and not reader.thread.is_alive()


def test_stream_reader_cancels_without_leaving_thread():
    class Capture:
        released=False
        def read(self): time.sleep(.01); return True,np.zeros((32,32,3),np.uint8)
        def release(self): self.released=True
    capture=Capture(); stop=threading.Event()
    reader=StreamReader("test-source",30,2,stop,opener=lambda _:capture)
    values=reader.samples(2); next(values); stop.set(); values.close()
    assert capture.released and not reader.thread.is_alive()


def test_group_leakage_is_rejected(tmp_path):
    a=tmp_path/"a.mp4"; b=tmp_path/"b.mp4"; a.touch(); b.touch()
    manifest=tmp_path/"manifest.csv"
    with manifest.open("w",newline="") as handle:
        writer=csv.writer(handle); writer.writerow(["path","label","group","split"])
        writer.writerow(["a.mp4","real","same-subject","train"])
        writer.writerow(["b.mp4","fake","same-subject","test"])
    with pytest.raises(ValueError,match="leakage"):
        read_manifest(manifest)


def test_real_ffmpeg_http_stream_transport():
    """Exercise actual HTTP media decoding, not a mocked VideoCapture."""
    import cv2
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                for i in range(20):
                    frame=np.full((128,128,3),i*10,np.uint8)
                    _,jpeg=cv2.imencode(".jpg",frame)
                    data=jpeg.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(data)).encode()+b"\r\n\r\n"+data+b"\r\n")
                    self.wfile.flush(); time.sleep(.1)
            except (BrokenPipeError,ConnectionResetError):
                pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    stop=threading.Event()
    reader=StreamReader(f"http://127.0.0.1:{server.server_port}/live",10,20,stop)
    try:
        values=list(reader.samples(5))
        assert len(values)>0
        assert all(frame.shape == (128,128,3) for _,frame in values)
        assert reader.done and not reader.thread.is_alive()
    finally:
        stop.set();server.shutdown();server.server_close();thread.join(timeout=1)
