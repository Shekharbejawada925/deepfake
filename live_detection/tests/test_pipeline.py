import cv2
import numpy as np
import pytest
from deepfake.vision import FaceExtractor, FaceFrame, iter_video, normalize_crops
from deepfake.inference import analyze_window, summarize
from deepfake.contract import default_metadata, validate_metadata
from conftest import StubDetector, StubExtractor


def samples(n=10):
    return [(i/10,np.zeros((128,128,3),np.uint8)) for i in range(n)]


def test_incomplete_window_never_produces_a_verdict():
    detector=StubDetector()
    result=analyze_window(samples(3),detector,StubExtractor(detector.meta))
    assert result["p_fake"] is None and result["label"] == "INCONCLUSIVE"
    assert detector.calls == 0


@pytest.mark.parametrize("reason",["no_face","multiple_faces","blurred_face","face_too_small","poor_lighting"])
def test_bad_face_evidence_cannot_be_called_real(reason):
    class Bad(StubExtractor):
        def extract(self,frame): return FaceFrame(reason=reason)
    detector=StubDetector()
    result=analyze_window(samples(),detector,Bad(detector.meta))
    assert result["p_fake"] is None and reason in result["reasons"]
    assert detector.calls == 0


def test_sampling_gaps_do_not_join_unrelated_frames():
    detector=StubDetector(); data=samples(); data[-1]=(8.,data[-1][1])
    result=analyze_window(data,detector,StubExtractor(detector.meta))
    assert "irregular_sampling" in result["reasons"] and detector.calls == 0


def test_track_jump_abstains():
    class Jump(StubExtractor):
        count=0
        def extract(self,frame):
            self.count+=1
            return FaceFrame(np.zeros((112,112,3),np.uint8),(.05,.05,.2,.2) if self.count<5 else (.7,.7,.2,.2))
    detector=StubDetector()
    result=analyze_window(samples(),detector,Jump(detector.meta))
    assert "scene_or_track_change" in result["reasons"] and detector.calls == 0


@pytest.mark.parametrize("score,label",[(.05,"LIKELY REAL"),(.5,"INCONCLUSIVE"),(.95,"LIKELY FAKE")])
def test_uncertainty_interval(score,label):
    detector=StubDetector(score)
    result=analyze_window(samples(),detector,StubExtractor(detector.meta))
    assert result["label"] == label


def test_partial_coverage_cannot_certify_whole_clip():
    detector=StubDetector()
    real=analyze_window(samples(),detector,StubExtractor(detector.meta))
    short=analyze_window(samples(2),detector,StubExtractor(detector.meta))
    assert summarize([real,short])["label"] == "INCONCLUSIVE"
    assert summarize([real])["label"] == "LIKELY REAL"


def test_bgr_is_converted_to_rgb_before_normalization():
    meta=default_metadata(10,10); meta["min_blur"]=0
    extractor=FaceExtractor(meta)
    class FixedCascade:
        def detectMultiScale(self,*args,**kwargs): return [(16,16,80,80)]
    extractor.cascade=FixedCascade()
    bgr=np.full((128,128,3),(10,100,220),np.uint8)
    face=extractor.extract(bgr)
    assert tuple(face.crop[40,40]) == (220,100,10)
    normalized=normalize_crops([face.crop],meta)
    assert normalized.shape == (1,3,112,112)
    assert normalized[0,0,40,40] == pytest.approx((220/255-.485)/.229)


def test_short_video_decodes_without_modulo_by_zero(tmp_path):
    path=tmp_path/"short.avi"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(64,64))
    assert writer.isOpened()
    for _ in range(3): writer.write(np.zeros((64,64,3),np.uint8))
    writer.release()
    assert len(list(iter_video(path,10))) == 3


@pytest.mark.parametrize("key,value",[("classes",["real","real"]),("sample_fps",float("nan")),("color_space","BGR"),("architecture","legacy"),("training_complete","yes")])
def test_checkpoint_contract_rejects_ambiguous_metadata(key,value):
    meta=default_metadata(); meta[key]=value
    with pytest.raises(ValueError): validate_metadata(meta)


def test_25fps_video_samples_at_ten_fps_without_drift(tmp_path):
    path = tmp_path / "25fps.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 64))
    for _ in range(75):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    frames = list(iter_video(path, 10))
    assert len(frames) == 30
    assert frames[-1][0] == pytest.approx(2.92, abs=.01)


def test_legacy_preserves_bgr_crop_order():
    from deepfake.contract import legacy_metadata
    meta = legacy_metadata("0" * 64); meta["min_blur"] = 0
    extractor = FaceExtractor(meta)
    class FixedCascade:
        def detectMultiScale(self, *args, **kwargs): return [(16, 16, 80, 80)]
    extractor.cascade = FixedCascade()
    face = extractor.extract(np.full((128, 128, 3), (10, 100, 220), np.uint8))
    assert tuple(face.crop[40, 40]) == (10, 100, 220)


@pytest.mark.parametrize("key", ["training_complete", "calibrated", "label_mapping_verified"])
def test_legacy_cannot_claim_training_or_label_verification(key):
    from deepfake.contract import legacy_metadata
    meta = legacy_metadata("0" * 64); meta[key] = True
    with pytest.raises(ValueError): validate_metadata(meta)


def test_partial_legacy_window_scores_one_final_face():
    from deepfake.contract import legacy_metadata
    detector = StubDetector(); detector.meta = legacy_metadata("0" * 64)
    detector.compatibility_mode = True
    result = analyze_window(samples(3), detector, StubExtractor(detector.meta))
    assert result["p_fake"] is not None
    assert result["score_scope"] == "final sampled face only"


def test_partial_legacy_window_still_rejects_bad_faces():
    from deepfake.contract import legacy_metadata
    detector = StubDetector(); detector.meta = legacy_metadata("0" * 64)
    class Bad(StubExtractor):
        def extract(self, frame): return FaceFrame(reason="no_face")
    result = analyze_window(samples(3), detector, Bad(detector.meta))
    assert result["p_fake"] is None and detector.calls == 0
