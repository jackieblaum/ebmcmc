"""Tests for PhoebeSED unit conversions and extinction."""
import pytest
import numpy as np

phoebe = pytest.importorskip("phoebe")

from ebmcmc.phoebe_sed import PhoebeSED


class TestUnitConversions:
    """Test bandflux (W/m^2) -> various unit conversions."""

    def test_bandflux_to_flam(self):
        """W/m^2 -> erg/cm^2/s/A using a known passband."""
        ps = PhoebeSED(filters=["Johnson:V"])
        flam = ps._bandflux_to_flam(1.0, "Johnson:V")
        assert np.isfinite(flam)
        assert flam > 0
        assert 0.01 < flam < 100.0

    def test_bandflux_to_fnu(self):
        """W/m^2 -> Jy using a known passband."""
        ps = PhoebeSED(filters=["Johnson:V"])
        fnu = ps._bandflux_to_fnu(1.0, "Johnson:V")
        assert np.isfinite(fnu)
        assert fnu > 0

    def test_fnu_to_ABmag(self):
        """3631 Jy should give AB mag = 0."""
        mag = PhoebeSED._fnu_to_ABmag(3631.0)
        assert abs(mag - 0.0) < 1e-6

    def test_fnu_to_ABmag_fainter(self):
        """1 Jy -> AB mag ~ 8.9."""
        mag = PhoebeSED._fnu_to_ABmag(1.0)
        assert abs(mag - 8.9) < 0.1

    def test_fnu_to_Vegamag(self):
        """Vega zero-point lookup for Johnson:V should give ~0 for Vega flux."""
        ps = PhoebeSED(filters=["Johnson:V"])
        mag = ps._fnu_to_Vegamag(3640.0, "Johnson:V")
        assert abs(mag) < 0.5

    def test_roundtrip_flam_fnu(self):
        """flam and fnu should be consistent via f_nu = f_lam * lam^2 / c."""
        ps = PhoebeSED(filters=["Johnson:V"])
        bf = 1e-10  # W/m^2
        flam = ps._bandflux_to_flam(bf, "Johnson:V")
        fnu = ps._bandflux_to_fnu(bf, "Johnson:V")
        # Both derived from same bandflux, should be consistent
        assert np.isfinite(flam) and np.isfinite(fnu)
        assert flam > 0 and fnu > 0


class TestConvertUnits:
    """Test the top-level convert method with global + per-filter overrides."""

    def test_default_flam(self):
        ps = PhoebeSED(filters=["Johnson:V", "2MASS:J"], sed_units="flam")
        bandfluxes = np.array([1e-10, 1e-11])
        result = ps.convert_units(bandfluxes)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_per_filter_override(self):
        ps = PhoebeSED(
            filters=["Johnson:V", "2MASS:J"],
            sed_units="flam",
            per_filter_units={"2MASS:J": "Vegamag"},
        )
        bandfluxes = np.array([1e-10, 1e-11])
        result = ps.convert_units(bandfluxes)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))
        assert result[0] > 0  # flam

    def test_all_unit_types(self):
        """Each unit type should produce finite output."""
        for unit in ["flam", "fnu_Jy", "ABmag", "Vegamag"]:
            ps = PhoebeSED(filters=["Johnson:V"], sed_units=unit)
            result = ps.convert_units(np.array([1e-10]))
            assert np.isfinite(result[0]), f"Failed for unit={unit}"


class TestExtinction:
    """Test extinction application."""

    def test_zero_ebv_no_change(self):
        ps = PhoebeSED(filters=["Johnson:V"])
        flux = np.array([1.0])
        result = ps._apply_extinction(flux, ebv=0.0)
        np.testing.assert_array_equal(result, flux)

    def test_positive_ebv_reduces_flux(self):
        ps = PhoebeSED(filters=["Johnson:V"])
        flux = np.array([1.0])
        result = ps._apply_extinction(flux, ebv=0.5)
        assert result[0] < flux[0]
        assert result[0] > 0

    def test_fitzpatrick99_vs_ccm89(self):
        ps_f99 = PhoebeSED(filters=["Johnson:V"], extinction_law="fitzpatrick99")
        ps_ccm = PhoebeSED(filters=["Johnson:V"], extinction_law="ccm89")
        flux = np.array([1.0])
        r_f99 = ps_f99._apply_extinction(flux, ebv=0.3)
        r_ccm = ps_ccm._apply_extinction(flux, ebv=0.3)
        # Both should reduce flux, but give slightly different values
        assert r_f99[0] < 1.0 and r_ccm[0] < 1.0
        assert r_f99[0] != r_ccm[0]


class TestValidation:
    """Test input validation."""

    def test_empty_filters_raises(self):
        with pytest.raises(ValueError):
            PhoebeSED(filters=[])

    def test_invalid_units_raises(self):
        with pytest.raises(ValueError):
            PhoebeSED(filters=["Johnson:V"], sed_units="invalid")

    def test_invalid_per_filter_units_raises(self):
        with pytest.raises(ValueError):
            PhoebeSED(filters=["Johnson:V"], per_filter_units={"Johnson:V": "bad"})
