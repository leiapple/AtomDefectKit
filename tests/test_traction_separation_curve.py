import numpy as np
import pytest
from ase.build import bulk

from atomdefectkit import TractionSeparationCurve, TractionSeparationWorkflow


def test_traction_separation_curve_reports_peak_and_work_of_separation():
    curve = TractionSeparationCurve.from_arrays(
        separation=[0.0, 1.0, 2.0],
        energy=[0.0, 2.0, 1.0],
        area=2.0,
    )

    assert np.allclose(curve.traction(), [2.0, 0.5, -1.0])
    assert curve.peak_traction() == 2.0
    assert curve.work_of_separation() == 1.0


def test_traction_separation_workflow_rejects_unsupported_surface():
    atoms = bulk("W", "bcc", a=3.2, cubic=True)

    with pytest.raises(ValueError, match="surface_index must be one of"):
        TractionSeparationWorkflow(
            atoms=atoms,
            calculator=None,
            surface_index=(2, 0, 0),
        )
