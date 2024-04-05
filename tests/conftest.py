import pathlib

from typing import Tuple

import pytest


file_path = pathlib.Path(__file__)
test_folder = file_path.parent.absolute()
proj_folder = test_folder.parent.absolute()



def py_files() -> Tuple[pathlib.Path]:
    return tuple(proj_folder.glob("wk*.py"))


def pytest_generate_tests(metafunc):
    if "py_file" in metafunc.fixturenames:
        metafunc.parametrize("py_file", py_files())
