import pytest

import atomdefectkit


EXPECTED_MODELS = {
    "7net",
    "chgnet",
    "fairchem",
    "grace",
    "mace",
    "mattersim",
    "nequip",
    "nequix",
    "ocp",
    "pace",
    "upet",
}


def test_available_models_includes_registered_backends():
    assert set(atomdefectkit.available_models()) == EXPECTED_MODELS


def test_available_model_metadata_imports_lazy_registrations():
    metadata = atomdefectkit.available_model_metadata()

    assert EXPECTED_MODELS.issubset(metadata)
    assert metadata["pace"]["potential_file"]["required"] is True
    assert metadata["fairchem"]["model_task"]["default"] == "omat"
    assert metadata["ocp"]["model_name"]["default"] == "EquiformerV2-31M-S2EF-OC20-All+MD"
    assert "eqV3-omat24-mptrj-salex_gradient" in metadata["ocp"]["model_name"]["choices"]
    assert "GRACE-2L-SMAX-OMAT-large" in metadata["grace"]["model_name"]["choices"]


def test_load_model_reports_unknown_model():
    with pytest.raises(ModuleNotFoundError):
        atomdefectkit.load_model("does-not-exist")
