"""Every R export at spiDE e31e744 is mapped in docs/r-to-python.md, and the mapping is real."""

import importlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R_REPO = ROOT.parent / "spiDE"
DOC = ROOT / "docs" / "r-to-python.md"


def _r_exports() -> list[str]:
    """Names in ``export()`` and ``exportClasses()`` of the R NAMESPACE (methods are the same names)."""
    ns = subprocess.run(
        ["git", "-C", str(R_REPO), "show", "e31e744:NAMESPACE"],
        check=True, capture_output=True, text=True,
    ).stdout
    return re.findall(r"^export(?:Classes)?\(([^)]+)\)$", ns, flags=re.M)


def _rows() -> dict[str, str]:
    """Map the first-column R name to the second-column Python cell of every row under "Exports"."""
    rows = {}
    text = DOC.read_text()
    for line in text[text.index("## Exports"):].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 2 and cells[0].startswith("`"):
            rows[cells[0].strip("`")] = cells[1]
    return rows


def _resolve(dotted: str):
    """Import the longest module prefix, then getattr the rest (e.g. ``spidepy.pl.style``)."""
    parts = dotted.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


@pytest.mark.skipif(not R_REPO.exists(), reason="../spiDE not available")
def test_every_r_export_mapped():
    rows = _rows()
    missing = [name for name in _r_exports() if name not in rows]
    assert not missing, f"R exports missing from docs/r-to-python.md: {missing}"


def test_python_names_in_mapping_resolve():
    rows = _rows()
    assert rows, "no mapping rows parsed"
    for r_name, cell in rows.items():
        refs = re.findall(r"`([A-Za-z_][\w.]*)`", cell)
        assert refs, f"{r_name}: no Python name given"
        # public rows live in spidepy, spidepy.pl or spidepy.datasets
        assert all(re.fullmatch(r"spidepy(\.(pl|datasets))?\.\w+", ref) for ref in refs), (r_name, refs)
        for ref in refs:
            assert _resolve(ref) is not None, f"{r_name}: {ref} not importable"


def test_public_api_is_documented():
    """Everything in ``spidepy.__all__`` (and ``pl.__all__``) appears in docs/api.md."""
    import spidepy

    api = (ROOT / "docs" / "api.md").read_text()
    names = [n for n in spidepy.__all__ if n != "pl"] + [f"pl.{n}" for n in spidepy.pl.__all__]
    missing = [n for n in names if not re.search(rf"^\s+{re.escape(n)}$", api, flags=re.M)]
    assert not missing, f"missing from docs/api.md: {missing}"
