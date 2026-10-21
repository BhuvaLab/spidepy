"""Load reference values written by tests/ref/make_*.R (rhdf5 HDF5 files)."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

REF_DIR = Path(__file__).parent / "ref" / "data"


def _decode(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind in ("S", "O"):
        return np.array(
            [s.decode() if isinstance(s, bytes) else s for s in a.ravel()], dtype=str
        ).reshape(a.shape)
    return a


def _levels(obj) -> list[str] | None:
    if "levels" not in obj.attrs:
        return None
    return [
        s.decode() if isinstance(s, bytes) else str(s) for s in np.atleast_1d(obj.attrs["levels"])
    ]


def _read(obj):
    if isinstance(obj, h5py.Group):
        return {k: _read(obj[k]) for k in obj}
    lev = _levels(obj)
    a = obj[()]
    if isinstance(a, bytes):
        return a.decode()
    a = np.asarray(a)
    if lev is not None:  # R factor: character codes + levels -> Categorical in R's level order
        return pd.Categorical(_decode(a.ravel()), categories=lev)
    if a.ndim == 1 and a.size == 1:
        a = a.reshape(())  # length-1 R vector -> scalar (1x1 matrices are 2-D and stay arrays)
    if a.ndim == 0:
        return _decode(a.reshape(1))[0].item() if a.dtype.kind in "SO" else a.item()
    # rhdf5 writes R's column-major array reversed; h5py sees dims reversed, so transpose back.
    return _decode(a.T if a.ndim > 1 else a)


def load_ref(topic: str) -> dict:
    """Read ``tests/ref/data/<topic>.h5`` with R's array shapes.

    Length-1 R vectors become Python scalars; use ``np.atleast_1d`` if you need an array.
    1x1 matrices keep shape (1, 1). R ``NA`` becomes ``nan`` for doubles (an integer or logical
    ``NA`` arrives as the integer minimum, so write such columns as doubles). R factors become
    ``pandas.Categorical`` with R's levels; logicals arrive as int8.
    """
    with h5py.File(REF_DIR / f"{topic}.h5", "r") as f:
        return {k: _read(f[k]) for k in f}


def load_spe(topic: str) -> ad.AnnData:
    """Build the AnnData for a SpatialExperiment fixture written by ``write_spe_h5ad()``.

    ``X`` and ``layers["counts"]`` hold the counts (cells x genes) in R's storage type,
    ``obs`` the colData columns in R's order and types (factors keep R's levels),
    ``obsm["spatial"]`` the spatial coordinates and ``obsm[<name>]`` any reducedDims. The
    dimensions R reported are kept in ``uns["r_dim"]`` (genes, cells) and the generating call
    in ``uns["r_call"]``.
    """
    d = load_ref(topic)
    cells = np.atleast_1d(d["cell_names"]).astype(str)
    genes = np.atleast_1d(d["gene_names"]).astype(str)
    counts = np.asarray(d["counts"])
    # R factors arrive as pd.Categorical (levels kept); atleast_1d would flatten them to arrays
    obs = pd.DataFrame(
        {
            c: v if isinstance(v := d["obs"][c], pd.Categorical) else np.atleast_1d(v)
            for c in np.atleast_1d(d["obs_columns"]).astype(str)
        },
        index=pd.Index(cells),
    )
    a = ad.AnnData(
        X=counts.copy(),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes)),
        layers={"counts": counts.copy()},
    )
    a.obsm["spatial"] = np.asarray(d["spatial"], dtype=float)
    for name in np.atleast_1d(d.get("reduced_names", [])).astype(str):
        a.obsm[name] = np.asarray(d["reduced"][name], dtype=float)
    a.uns["r_dim"] = np.atleast_1d(d["dim"]).astype(int)
    a.uns["r_call"] = str(d.get("call", ""))
    return a


def write_bundled_dataset() -> Path:
    """Write ``src/spidepy/datasets/toySpiDE.h5ad`` from the R ``toySpiDE`` fixture."""
    out = Path(__file__).parents[1] / "src" / "spidepy" / "datasets" / "toySpiDE.h5ad"
    load_spe("toySpiDE").write_h5ad(out)
    return out


if __name__ == "__main__":
    print(write_bundled_dataset())
