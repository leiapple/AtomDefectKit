import re

from atomdefectkit.model_errors import format_model_error, format_unknown_model_error
from atomdefectkit.registry import MODEL_METADATA, MODEL_REGISTRY, register_model
from atomdefectkit.utils.progress import progress


def test_format_unknown_model_error_lists_available_models():
    message = format_unknown_model_error("mystery", ["mace", "upet"])

    assert "Unknown model 'mystery'." in message
    assert "- 'mace'" in message
    assert "- 'upet'" in message


def test_format_unknown_model_error_handles_empty_registry():
    message = format_unknown_model_error("mystery", [])

    assert "Unknown model 'mystery'." in message
    assert "(none found)" in message


def test_format_model_error_includes_known_metadata():
    MODEL_METADATA["test-model"] = {"alpha": {"default": 1}}
    try:
        message = format_model_error("test-model", {"alpha": 2}, ValueError("boom"))
    finally:
        MODEL_METADATA.pop("test-model", None)

    assert "Model: test-model" in message
    assert "Error type: ValueError" in message
    assert '"default": 1' in message


def test_register_model_stores_lowercase_name_and_metadata():
    @register_model("Temp-Model", metadata={"beta": {"default": 3}})
    def builder(params, device):
        return params, device

    try:
        assert MODEL_REGISTRY["temp-model"] is builder
        assert MODEL_METADATA["temp-model"] == {"beta": {"default": 3}}
    finally:
        MODEL_REGISTRY.pop("temp-model", None)
        MODEL_METADATA.pop("temp-model", None)


def test_progress_prints_timestamped_message(capsys):
    progress("hello world")

    output = capsys.readouterr().out.strip()
    assert output.endswith("hello world")
    assert re.match(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] hello world$", output)
