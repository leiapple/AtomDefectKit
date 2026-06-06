import numpy as np

from atomdefectkit.TractionSeparation import _integrate_trapezoid


def test_integrate_trapezoid_matches_numpy_trapezoid():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 2.0, 0.0])

    assert _integrate_trapezoid(y, x) == np.trapezoid(y, x)
