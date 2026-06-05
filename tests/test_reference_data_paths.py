from ase import Atoms
import numpy as np

from atomdefectkit.neb import BCCScrewDislocPeierlsBarrier
from atomdefectkit.stacking_fault import StackingFaultWorkflow


def test_stacking_fault_reference_curve_lookup_finds_repo_data(tmp_path):
    atoms = Atoms("V", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True)
    workflow = StackingFaultWorkflow(
        atoms=atoms,
        calculator=None,
        formula="V",
        working_dir=tmp_path,
    )

    path_110 = workflow._reference_curve_path((1, -1, 0))
    path_112 = workflow._reference_curve_path((1, 1, 2))

    assert path_110 is not None
    assert path_110.name == "V_110.csv"
    assert path_112 is not None
    assert path_112.name == "V_112.csv"


def test_peierls_reference_curve_lookup_finds_repo_data(tmp_path):
    atoms = Atoms("V", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True)
    workflow = BCCScrewDislocPeierlsBarrier(
        initial_config=atoms.copy(),
        final_config=atoms.copy(),
        working_dir=tmp_path,
    )

    path = workflow._reference_barrier_path("V")

    assert path is not None
    assert path.name == "V_bcc_VASP.csv"


def test_stacking_fault_112_reference_curve_is_reversed(tmp_path):
    atoms = Atoms("V", positions=[[0.0, 0.0, 0.0]], cell=[1.0, 1.0, 1.0], pbc=True)
    workflow = StackingFaultWorkflow(
        atoms=atoms,
        calculator=None,
        formula="V",
        working_dir=tmp_path,
    )

    reference_curve = np.array(
        [
            [0.0, 1.0],
            [0.5, 2.0],
            [1.0, 3.0],
        ]
    )

    x_values, y_values = workflow._format_reference_curve(reference_curve, (1, 1, 2))

    assert np.allclose(x_values, [0.0, 0.5, 1.0])
    assert np.allclose(y_values, [3000.0, 2000.0, 1000.0])
