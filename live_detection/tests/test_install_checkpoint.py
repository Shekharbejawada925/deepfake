import hashlib
import json
import zipfile
import pytest
from tools.install_checkpoint import install


def profile(tmp_path, value=b"known weights"):
    models = tmp_path / "models"
    models.mkdir()
    (models / "legacy_detector.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(value).hexdigest()}))
    return models


def test_install_from_saved_zip_checks_hash_and_is_repeatable(tmp_path):
    models = profile(tmp_path)
    source = tmp_path / "saved.zip"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("Deepfake_Streaming/models/legacy_detector.pt", b"known weights")
        z.writestr("../../unexpected.txt", b"never extracted")
    target = install(source, models)
    assert target.read_bytes() == b"known weights"
    assert install(source, models) == target
    assert not (tmp_path / "unexpected.txt").exists()
    assert not list(models.glob("*.pending"))


def test_wrong_checkpoint_is_rejected_without_creating_model(tmp_path):
    models = profile(tmp_path)
    source = tmp_path / "bad.pt"; source.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        install(source, models)
    assert not (models / "legacy_detector.pt").exists()
    assert not list(models.glob("*.pending"))


def test_installer_preserves_an_existing_different_model(tmp_path):
    models = profile(tmp_path)
    target = models / "legacy_detector.pt"; target.write_bytes(b"different")
    with pytest.raises(ValueError, match="already exists"):
        install(tmp_path / "not-needed.pt", models)
    assert target.read_bytes() == b"different"
