# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""CPU-only source checks for experimental API containment."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["add_bias_residual", "bias_silu_multiply"])
def test_bias_entry_point_is_thin_and_explicitly_experimental(name: str) -> None:
    # Given the real public module source, without importing optional dependencies.
    path = Path(__file__).resolve().parents[2] / "flashinfer/diffusion_ops/h3_bridge.py"
    module = ast.parse(path.read_text())
    functions = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    # When locating the requested public API.
    assert name in functions
    function = functions[name]
    # Then its decorator opts in and the backend import stays inside the function.
    assert any(
        isinstance(node, ast.Name) and node.id == "flashinfer_experimental_api"
        for node in function.decorator_list
    )
    body = (
        function.body[1:] if ast.get_docstring(function) is not None else function.body
    )
    assert (
        len(body) == 2
        and isinstance(body[0], ast.ImportFrom)
        and isinstance(body[1], ast.Return)
    )
    assert body[0].module == "experimental.h3_bridge.bias_epilogue"
    assert not any(
        isinstance(node, ast.ImportFrom)
        and "experimental.h3_bridge" in (node.module or "")
        for node in module.body
    )


def test_bias_backend_imports_optional_dependencies_only_on_demand() -> None:
    # Given the actual implementation source.
    path = (
        Path(__file__).resolve().parents[2]
        / "flashinfer/experimental/h3_bridge/bias_epilogue.py"
    )
    assert path.exists()
    module = ast.parse(path.read_text())
    # When examining executable module-level imports.
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    # Then Torch/Triton/CUDA are not required merely to import the backend.
    for node in imports:
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
        assert not any(
            name.split(".")[0] in {"torch", "triton", "cuda"} for name in names
        )
