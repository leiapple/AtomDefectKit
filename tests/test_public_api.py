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


def test_load_model_reports_unknown_model():
    with pytest.raises(ModuleNotFoundError):
        atomdefectkit.load_model("does-not-exist")
