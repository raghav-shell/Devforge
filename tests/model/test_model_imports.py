"""Import hygiene for submission/doom_agent/models (OWNERSHIP rule 4, MODULE_CONTRACTS 3.1)."""

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "submission" / "doom_agent" / "models"

FORBIDDEN_ROOTS = {"training", "env", "doom_env", "vizdoom", "submission"}
ALLOWED_ROOTS = {"__future__", "dataclasses", "typing", "math", "torch", "numpy", "doom_agent"}
NUMPY_ONLY = {"action_codec.py", "__init__.py"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add("<relative>")
            elif node.module:
                roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            roots.add("<dynamic>")
    return roots


def test_models_have_no_forbidden_imports():
    files = sorted(MODELS.rglob("*.py"))
    assert {f.name for f in files} >= {
        "action_codec.py", "distributions.py", "config.py", "encoders.py", "policy.py", "io.py", "__init__.py"
    }
    for f in files:
        roots = _imports(f)
        assert not (roots & FORBIDDEN_ROOTS), (f, roots & FORBIDDEN_ROOTS)
        assert roots <= ALLOWED_ROOTS, (f, roots - ALLOWED_ROOTS)
        if f.name in NUMPY_ONLY:
            assert "torch" not in roots, f


def test_codec_import_is_torch_free():
    code = (
        "import sys; import doom_agent.models.action_codec as ac; "
        "assert 'torch' not in sys.modules, 'torch leaked'; print(ac.NUM_LOGITS)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60,
        env={"PYTHONPATH": str(ROOT / "submission"), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "15"


def test_no_privileged_access_in_models():
    for f in MODELS.rglob("*.py"):
        assert "privileged_state" not in f.read_text(), f
