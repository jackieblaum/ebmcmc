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
    b, sed_obj, phoebe_sed_obj = _build_template_bundle(data_dict, model_phases=None, use_ellc=False)
    assert "lc01" in b.datasets
    assert sed_obj is None
    assert phoebe_sed_obj is None


@pytest.mark.phoebe
def test_build_template_bundle_phoebe_sed():
    """Smoke test: _build_template_bundle with sed_method='phoebe'."""
    from ebmcmc.loglike import _build_template_bundle

    data_dict = {
        "lc01": {
            "data": np.ones(50),
            "sigmas": 0.01 * np.ones(50),
            "times": np.linspace(0, 10, 50),
            "phases": np.linspace(0, 1, 50),
            "passband": "TESS:T",
        },
        "sed": {
            "filters": ["Johnson:V", "2MASS:J"],
            "fluxes": np.array([1e-12, 1e-13]),
            "flux_errs": np.array([1e-13, 1e-14]),
            "dist": 500.0,
            "ebv": 0.1,
        },
    }
    b, sed_obj, phoebe_sed_obj = _build_template_bundle(
        data_dict, model_phases=None, use_ellc=False, sed_method="phoebe"
    )
    assert "lc01" in b.datasets
    assert sed_obj is None
    assert phoebe_sed_obj is not None
    assert len(phoebe_sed_obj.filters) == 2


@pytest.mark.phoebe
def test_ebmcmc_accepts_sed_method_phoebe():
    """EBMCMC constructor accepts sed_method='phoebe'."""
    from ebmcmc import EBMCMC

    b = phoebe.default_binary()
    times = np.linspace(0, 10, 50)
    fluxes = 1.0 + 0.01 * np.random.randn(len(times))
    sigmas = 0.01 * np.ones(len(times))
    b.add_dataset("lc", times=times, fluxes=fluxes, sigmas=sigmas, dataset="lc01")
    b.set_value("passband@lc01", "TESS:T")

    sed = {
        "RA": 180.0, "DEC": 30.0, "dist": 500.0,
        "filters": ["Johnson:V", "2MASS:J"],
        "fluxes": np.array([1e-12, 1e-13]),
        "flux_errs": np.array([1e-13, 1e-14]),
    }

    eb = EBMCMC(b, trace_dir="/tmp/ebmcmc_test", sed=sed,
                sed_method="phoebe", sed_units="flam",
                new_run_dir="test_phoebe_sed")
    assert eb.sed_method == "phoebe"
    assert eb.sed_units == "flam"
    assert "sed" in eb.data_dict
