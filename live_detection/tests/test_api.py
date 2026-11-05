import base64
import io
import unittest
from PIL import Image
from app import app, predictor

class APITests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()
    def test_live_page(self):
        self.assertEqual(self.client.get("/live").status_code,200)
    def test_status(self):
        response=self.client.get("/api/live/status")
        self.assertEqual(response.status_code,200)
        self.assertIn("ready",response.get_json())
    def test_missing_frames(self):
        self.assertEqual(self.client.post("/api/live/predict",json={"frames":[]}).status_code,400)
    def test_bad_jpeg(self):
        frames=["data:image/jpeg;base64,invalid"]*8
        self.assertEqual(self.client.post("/api/live/predict",json={"frames":frames}).status_code,400)
    def test_model_unavailable(self):
        image=Image.new("RGB",(64,64),color="white")
        buf=io.BytesIO()
        image.save(buf,"JPEG")
        payload="data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()
        if not predictor.ready:
            self.assertEqual(self.client.post("/api/live/predict",json={"frames":[payload]*8}).status_code,503)

if __name__=="__main__":
    unittest.main()
