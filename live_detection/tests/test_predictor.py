import math
import unittest
from predictor import Predictor

class PredictorTests(unittest.TestCase):
    def test_inconclusive_face_quality(self):
        p=Predictor()
        p.fn=lambda frames: (_ for _ in ()).throw(ValueError("No face detected"))
        result=p.predict([object()]*8)
        self.assertEqual(result["label"],"Inconclusive")
        self.assertIsNone(result["fake_probability"])
    def test_invalid_score_rejected(self):
        p=Predictor()
        p.fn=lambda frames: math.nan
        with self.assertRaises(ValueError):
            p.predict([object()]*8)
    def test_boundaries(self):
        p=Predictor()
        for val,label in [(0.1,"Likely real"),(0.5,"Uncertain"),(0.9,"Likely fake")]:
            p.fn=lambda frames, val=val:val
            self.assertEqual(p.predict([object()]*8)["label"],label)

if __name__=="__main__":
    unittest.main()
