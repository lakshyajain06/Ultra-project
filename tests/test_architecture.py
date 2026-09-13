"""Dependency-boundary checks for the top-level source domains."""

import ast
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"


def imported_domains(package):
    domains = set()
    for path in (SRC / package).rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                domains.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                domains.add(node.module.partition(".")[0])
    return domains


def test_source_dependency_direction():
    assert imported_domains("data").isdisjoint({"sim", "learning"})
    assert imported_domains("sim").isdisjoint({"data", "learning"})
    assert "sim" not in imported_domains("learning")
