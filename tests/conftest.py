import pytest

from _ref import load_ref, load_spe


@pytest.fixture
def ref():
    """Callable returning the reference dictionary of a topic (``ref("rcompat")``)."""
    return load_ref


@pytest.fixture
def spe():
    """Callable building a fresh AnnData of a SpatialExperiment fixture (``spe("toyspe_default")``)."""
    return load_spe


@pytest.fixture
def toy_spide():
    """Fresh AnnData (cells x genes) of R's ``toySpiDE`` for each test."""
    return load_spe("toySpiDE")
