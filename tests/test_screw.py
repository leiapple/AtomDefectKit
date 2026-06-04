import numpy as np
import pytest

from atomdefectkit.screw import BCCScrewDislocation


def test_negative_c44_is_clamped_with_warning(tmp_path):
    cij = np.eye(6)
    cij[3, 3] = -5.0

    with pytest.warns(RuntimeWarning, match="Calculated C_44=.*negative"):
        workflow = BCCScrewDislocation(
            element="V",
            lattice_constant=3.0,
            elastic_constant=cij,
            calculator=None,
            working_dir=tmp_path,
        )

    assert workflow.cij[3, 3] == 0.001
    assert cij[3, 3] == -5.0
