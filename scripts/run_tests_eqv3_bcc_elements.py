import subprocess
import sys
from pathlib import Path

from atomdefectkit.utils.progress import progress


ELEMENT_A0 = [
    ("V", 2.997),
    ("Nb", 3.307),
    ("Ta", 3.163),
    ("Mo", 3.319),
    ("W", 3.185),
]


def main():
    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "run_tests_eqv3.py"

    for element, initial_a0 in ELEMENT_A0:
        working_dir = script_dir / f"Test_{element}_eqv3"
        progress(f"Running EqV3 workflow for {element} with initial a0={initial_a0} A")
        subprocess.run(
            [
                sys.executable,
                str(runner),
                "--element",
                element,
                "--initial-a0",
                str(initial_a0),
                "--working-dir",
                str(working_dir),
            ],
            cwd=script_dir,
            check=True,
        )
        progress(f"Finished EqV3 workflow for {element}")


if __name__ == "__main__":
    main()
