import ast
import logging
import pathlib

import pytest


file_path = pathlib.Path(__file__)
test_folder = file_path.parent.absolute()
proj_folder = test_folder.parent.absolute()

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def test_grammar(py_file:pathlib.Path):

    code = py_file.read_text(encoding="utf-8")

    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Syntax error in file: {py_file.relative_to(proj_folder)}\n{e}")


def test_module(py_file:pathlib.Path):

    code = py_file.read_text(encoding="utf-8")

    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                logger.info(f"Import: {alias.name}")
                if not(alias.name.startswith('numpy') or alias.name.startswith('scipy') or alias.name.startswith('matplotlib')):
                    pytest.fail(f"tried to import {alias.name} in {py_file.relative_to(proj_folder)}")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "numpy":
                pytest.fail(f"Import of numpy in {py_file.relative_to(proj_folder)}")
