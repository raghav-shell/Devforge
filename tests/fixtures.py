"""Pre-built test maps. Each layout is drawn in the test for readability; the matching WAD lives in
tests/fixtures/<sha1-of-layout>.wad."""

import hashlib
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_wad(grid: str) -> str:
    path = FIXTURES / f"{hashlib.sha1(grid.encode()).hexdigest()[:16]}.wad"
    if not path.exists():
        raise FileNotFoundError(f"missing test fixture {path.name} for layout:\n{grid}")
    return str(path)
