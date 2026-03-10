"""Integration tests requiring PHOEBE — skipped automatically if not installed."""
import pytest
import numpy as np

phoebe = pytest.importorskip("phoebe")


@pytest.mark.phoebe
def test_import_ebmcmc():
    """Verify that the package imports successfully."""
    from ebmcmc import EBMCMC
    assert EBMCMC is not None


@pytest.mark.phoebe
def test_build_template_bundle():
    """Smoke test for _build_template_bundle with minimal data."""
    from ebmcmc.loglike import _build_template_bundle

    data_dict = {
        "lc01": {
            "data": np.ones(50),
            "sigmas": 0.01 * np.ones(50),
            "times": np.linspace(0, 10, 50),
            "phases": np.linspace(0, 1, 50),
            "passband": "TESS:T",
        }
    }
    b, sed_obj = _build_template_bundle(data_dict, model_phases=None, use_ellc=False)
    assert "lc01" in b.datasets
    assert sed_obj is None
