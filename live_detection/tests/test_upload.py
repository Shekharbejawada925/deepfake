import io
import unittest
from app import app, predictor

class UploadTests(unittest.TestCase):
    def setUp(self):
        self.client=app.test_client()
        self.previous=predictor.fn

    def tearDown(self):
        predictor.fn=self.previous

    def test_no_checkpoint(self):
        predictor.fn=None
        response=self.client.post("/api/video/predict",
                                  data={"file":(io.BytesIO(b"not a video"),"clip.mp4")})
        self.assertEqual(response.status_code,503)

    def test_no_file(self):
        predictor.fn=lambda frames:0.5
        response=self.client.post("/api/video/predict")
        self.assertEqual(response.status_code,400)

    def test_reject_unknown_extension(self):
        predictor.fn=lambda frames:0.5
        response=self.client.post("/api/video/predict",
                                  data={"file":(io.BytesIO(b"not a video"),"clip.exe")})
        self.assertEqual(response.status_code,400)

if __name__=="__main__":
    unittest.main()
