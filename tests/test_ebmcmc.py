import pytest
from unittest.mock import MagicMock, patch
import phoebe
import pymc as pm
import numpy as np
import arviz as az
from ebmcmc import EBMCMC
import xarray as xr


@pytest.fixture
def mock_bundle():
    """Creates a mock PHOEBE bundle for testing."""
    bundle = MagicMock(spec=phoebe.Bundle)
    bundle.datasets = ["lc1", "rv1"]
    # Mock bundle values for initialization
    bundle.get_value.side_effect = lambda x: {
        "period@binary@component": 2.5,
        "mass@primary@component": 1.0,
        "mass@secondary@component": 0.8,
        "q@binary@component": 0.8,
        "incl@binary@component": 85,
        "asini@binary@component": 15.0,
        "requivsumfrac@binary@component": 0.2,
        "teffratio@binary@component": 0.9,
        "teff@secondary@component": 4500,
        "ecc@binary@component": 0.05,
        "per0@binary@component": 90,
        "fluxes@lc1@dataset": np.array([1.0, 1.1, 1.2]),
        "sigmas@lc1": np.array([0.01, 0.01, 0.01]),
        "times@lc1@dataset": np.array([0.1, 0.2, 0.3]),
        "rvs@primary@rv1@dataset": np.array([10.0, 12.0, 11.0]),
        "sigmas@rv1@primary": np.array([0.5, 0.5, 0.5]),
        "times@rv1@primary@dataset": np.array([0.1, 0.2, 0.3]),
        "fluxes@model@latest": np.array([1.1, 1.0, 1.2]),
    }.get(x, 0)
    return bundle

@pytest.fixture
def mock_trace():
    """Creates a mock PyMC trace for testing."""
    # Create DataArrays for each variable
    var1 = xr.DataArray([[1, 2], [3, 4]], dims=["chain", "draw"])
    var2 = xr.DataArray([[5, 6], [7, 8]], dims=["chain", "draw"])

    # Use InferenceData with individual arrays in the posterior group
    mock_trace = az.from_dict(posterior={"var1": var1, "var2": var2})

    return mock_trace

def test_initialize_ebmcmc(mock_bundle):
    """Test the initialization of EBMCMC class."""
    ebmcmc = EBMCMC(bundle=mock_bundle)
    assert ebmcmc.bundle == mock_bundle
    assert ebmcmc.eclipsing is True
    assert ebmcmc.ecc is True
    assert isinstance(ebmcmc.data_dict, dict)


def test_create_data_dict(mock_bundle):
    """Test the creation of data dictionary from the bundle."""
    ebmcmc = EBMCMC(bundle=mock_bundle)
    data_dict = ebmcmc.create_data_dict()
    assert "lc1" in data_dict
    assert "rv1" in data_dict
    assert np.array_equal(data_dict["lc1"]["data"], np.array([1.0, 1.1, 1.2]))


def test_define_model(mock_bundle):
    """Test defining the pymc model."""
    ebmcmc = EBMCMC(bundle=mock_bundle)
    ebmcmc.define_model()
    assert ebmcmc.model is not None

def test_sample(mock_bundle, mock_trace):
    """Test the sampling function with mocked pymc sample."""
    ebmcmc = EBMCMC(bundle=mock_bundle, trace_dir="test_trace")

    with patch("pymc.sample", return_value=mock_trace) as mock_sample:
        trace = ebmcmc.sample(ndraws=10, cores=1)
        assert mock_sample.called
        assert trace is not None


def test_save_trace(mock_bundle, mock_trace):
    """Test the trace saving functionality."""
    ebmcmc = EBMCMC(bundle=mock_bundle, trace_dir="test_trace")
    with patch("arviz.to_netcdf") as mock_save:
        ebmcmc.save_trace(mock_trace)
        assert mock_save.call_count == 2

# TODO: Figure out how to test these last two functions properly
# def test_load_full_trace_states(mock_bundle, mock_trace):
#     """Test loading the full trace states from files."""
#     ebmcmc = EBMCMC(bundle=mock_bundle, trace_dir="test_trace")
#     with patch(
#         "os.listdir", return_value=["trace_chain_0.nc", "trace_chain_1.nc"]
#     ), patch("arviz.from_netcdf", return_value=mock_trace):
#         trace = ebmcmc.load_full_trace_states()
#         assert trace is not None


# def test_posterior_predictive_checks(mock_bundle):
#     """Test posterior predictive checks."""
#     ebmcmc = EBMCMC(bundle=mock_bundle)
#     trace_mock = MagicMock()
#     with patch("matplotlib.pyplot.subplots"), patch.object(
#         ebmcmc, "bundle", mock_bundle
#     ):
#         ebmcmc.posterior_predictive_checks(trace_mock)
#         assert ebmcmc.bundle.run_compute.called
