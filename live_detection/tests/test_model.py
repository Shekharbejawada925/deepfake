import numpy as np
import pytest
import torch
from deepfake.model import VideoModel
from deepfake.inference import Detector
from deepfake.contract import default_metadata


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2); torch.manual_seed(123)
    return VideoModel(pretrained=False).eval()


def test_batch_members_do_not_change_each_others_prediction(model):
    # Regression for the original batch-as-time bug.
    features=torch.randn(2,4,2048)
    with torch.inference_mode():
        together=model.classify(features)
        alone=model.classify(features[:1])
    torch.testing.assert_close(together[:1],alone,atol=1e-6,rtol=1e-5)


def test_changing_early_frames_changes_temporal_result(model):
    features=torch.randn(1,5,2048)
    altered=features.clone(); altered[:,:-1] *= -1
    with torch.inference_mode():
        first=model.classify(features); second=model.classify(altered)
    assert not torch.allclose(first,second,atol=1e-6)


def test_full_encoder_and_lstm_forward_shape(model):
    with torch.inference_mode():
        result=model(torch.randn(1,10,3,112,112))
    assert result.shape == (1,2) and torch.isfinite(result).all()


def test_raw_legacy_weights_are_not_silently_reinterpreted(tmp_path):
    path=tmp_path/"legacy.pt"; torch.save({"linear1.weight":torch.zeros(2,2048)},path)
    detector=Detector(path)
    assert detector.ready is False and "Legacy" in detector.error


def test_untrained_checkpoint_is_refused(tmp_path):
    path=tmp_path/"untrained.pt"; torch.save({"metadata":default_metadata(),"state_dict":{}},path)
    detector=Detector(path)
    assert not detector.ready and "not a trained" in detector.error


def test_class_mapping_comes_from_metadata():
    detector=Detector("this_file_does_not_exist.pt")
    detector.ready=True; detector.meta=default_metadata(10,10)
    detector.meta["classes"]=["real","fake"]
    detector.device=torch.device("cpu")
    class StubModel:
        def encode(self,x): return torch.zeros(len(x),2048)
        def classify(self,x): return torch.tensor([[5.,-5.]])
    detector.model=StubModel()
    p,_=detector.score(np.zeros((10,112,112,3),np.uint8))
    assert p < .001


def test_legacy_graph_matches_original_and_ignores_earlier_frames():
    from deepfake.model import LegacyVideoModel
    legacy = LegacyVideoModel().eval()
    assert legacy.lstm.bias is False and legacy.lstm.batch_first is False
    features = torch.randn(1, 60, 2048)
    with torch.inference_mode():
        seq, _ = legacy.lstm(features)
        original = legacy.dp(legacy.linear1(seq[:, -1, :]))
        compatible = legacy.classify(features)
        shortcut = legacy.classify(features[:, -1:])
    torch.testing.assert_close(original, compatible)
    torch.testing.assert_close(original, shortcut, atol=1e-6, rtol=1e-5)


def test_legacy_profile_rejects_another_checkpoints_bytes(tmp_path):
    import json
    from deepfake.contract import legacy_metadata
    path = tmp_path / "legacy.pt"
    torch.save({"linear1.weight": torch.zeros(2, 2048)}, path)
    path.with_suffix(".json").write_text(json.dumps(legacy_metadata("0" * 64)))
    detector = Detector(path)
    assert not detector.ready and "does not match" in detector.error
