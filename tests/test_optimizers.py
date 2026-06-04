import pytest

from atomdefectkit.utils.optimizers import normalize_optimizer_name


@pytest.mark.parametrize(
    ("input_name", "expected_name"),
    [
        ("FIRE", "FIRE"),
        ("bfgs", "BFGS"),
        ("LBFGS", "LBFGS"),
        ("SciPyFminCG", "SCIPYFMINCG"),
        ("scipy_fmin_cg", "SCIPYFMINCG"),
        ("scipy-fmin-cg", "SCIPYFMINCG"),
    ],
)
def test_normalize_optimizer_name_accepts_supported_aliases(input_name, expected_name):
    assert normalize_optimizer_name(input_name) == expected_name


def test_normalize_optimizer_name_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match="Optimizer must be"):
        normalize_optimizer_name("not_an_optimizer")
