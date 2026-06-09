from pathlib import Path

import pytest

from atomdefectkit.models.ocp import EQV3_MODEL_URLS, _download_direct_checkpoint


def test_download_direct_checkpoint_uses_expected_url(monkeypatch, tmp_path):
    recorded = {}

    def fake_urlretrieve(url, destination):
        recorded["url"] = url
        recorded["destination"] = str(destination)
        Path(destination).write_text("checkpoint")

    monkeypatch.setattr("atomdefectkit.models.ocp.urlretrieve", fake_urlretrieve)

    checkpoint = _download_direct_checkpoint("eqV3-omat24-gradient", str(tmp_path))

    assert checkpoint == str(tmp_path / "omat24_gradient.pt")
    assert recorded["url"] == EQV3_MODEL_URLS["eqV3-omat24-gradient"]
    assert recorded["destination"] == str(tmp_path / "omat24_gradient.pt")


def test_download_direct_checkpoint_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown direct-checkpoint model name"):
        _download_direct_checkpoint("not-a-model", str(tmp_path))
