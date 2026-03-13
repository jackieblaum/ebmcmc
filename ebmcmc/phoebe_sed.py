"""
PHOEBE-based SED forward model for ebmcmc.

Computes passband-integrated fluxes using PHOEBE, applies extinction,
and converts to the user's target unit system (f_lambda, f_nu, AB mag,
or Vega mag).
"""
import numpy as np
import phoebe
import extinction
from astropy import units as u
import logging

logger = logging.getLogger(__name__)

# Vega zero-point fluxes in Jy for common passbands.
# Sources: Bessell et al. (1998), Cohen et al. (2003), Jarrett et al. (2011)
VEGA_ZP_JY = {
    "Johnson:U": 1810.0,
    "Johnson:B": 4260.0,
    "Johnson:V": 3640.0,
    "SDSS:u": 3631.0,
    "SDSS:g": 3631.0,
    "SDSS:r": 3631.0,
    "SDSS:i": 3631.0,
    "SDSS:z": 3631.0,
    "Pan-Starrs:g": 3631.0,
    "Pan-Starrs:r": 3631.0,
    "Pan-Starrs:i": 3631.0,
    "Pan-Starrs:z": 3631.0,
    "Pan-Starrs:y": 3631.0,
    "Pan-Starrs:w": 3631.0,
    "2MASS:J": 1594.0,
    "2MASS:H": 1024.0,
    "2MASS:Ks": 666.7,
    "WISE:W1": 309.54,
    "WISE:W2": 171.79,
    "GALEX:FUV": 3631.0,
    "GALEX:NUV": 3631.0,
    "Gaia:G": 3228.75,
    "Gaia:BP": 3552.01,
    "Gaia:RP": 2554.95,
}

C_LIGHT_M_S = 2.99792458e8


class PhoebeSED:
    """
    PHOEBE-based SED computation for use in MCMC forward models.

    Parameters
    ----------
    filters : list of str
        PHOEBE passband names for the observed filters (e.g., ["Johnson:V", "2MASS:J"]).
    sed_units : str
        Global target unit: "flam" (erg/cm^2/s/A), "fnu_Jy", "ABmag", "Vegamag".
    per_filter_units : dict, optional
        Per-filter unit overrides, e.g., {"2MASS:J": "Vegamag"}.
    sed_phases : list of float, optional
        Orbital phases at which to compute the SED. Default: [0.25].
    extinction_law : str
        "fitzpatrick99" or "ccm89".
    """

    VALID_UNITS = {"flam", "fnu_Jy", "ABmag", "Vegamag"}

    def __init__(self, filters, sed_units="flam", per_filter_units=None,
                 sed_phases=None, extinction_law="fitzpatrick99"):
        if not filters:
            raise ValueError("filters must be a non-empty list of passband names")
        self.filters = list(filters)
        if sed_units not in self.VALID_UNITS:
            raise ValueError(f"sed_units must be one of {self.VALID_UNITS}, got '{sed_units}'")
        self.sed_units = sed_units
        self.per_filter_units = per_filter_units or {}
        for k, v in self.per_filter_units.items():
            if v not in self.VALID_UNITS:
                raise ValueError(f"per_filter_units['{k}'] must be one of {self.VALID_UNITS}")
        self.sed_phases = sed_phases if sed_phases is not None else [0.25]
        self.extinction_law = extinction_law

        self._pb_cache = {}
        self._eff_wl_nm = np.full(len(self.filters), np.nan)
        self._bandwidth_m = np.full(len(self.filters), np.nan)
        self._pivot_wl_m = np.full(len(self.filters), np.nan)

        for i, filt in enumerate(self.filters):
            pb = phoebe.get_passband(filt)
            self._pb_cache[filt] = pb
            self._eff_wl_nm[i] = pb.effwl
            wl_m = np.asarray(pb.ptf_table["wl"], dtype=float)
            R = np.asarray(pb.ptf_table["fl"], dtype=float)
            ok = np.isfinite(wl_m) & np.isfinite(R) & (wl_m > 0) & (R > 0)
            wl_m, R = wl_m[ok], R[ok]
            if wl_m.size >= 3:
                self._bandwidth_m[i] = np.trapz(R, wl_m)
                num = np.trapz(wl_m * R, wl_m)
                den = np.trapz(R / wl_m, wl_m)
                self._pivot_wl_m[i] = np.sqrt(num / den) if den > 0 else np.nan

    def compute_sed(self, bundle, dist, ebv=0.0):
        """
        Compute SED by adding temporary LC datasets to the bundle.

        1. Disables all existing LC/RV datasets
        2. Adds temporary LC datasets for each observed filter
        3. Runs PHOEBE compute
        4. Extracts band fluxes, applies extinction, converts units
        5. Cleans up temp datasets and re-enables originals

        Parameters
        ----------
        bundle : phoebe.Bundle
            PHOEBE bundle with binary parameters already set (post-LC/RV compute).
        dist : float
            Distance in parsec.
        ebv : float
            E(B-V) color excess for extinction.

        Returns
        -------
        sed_model : ndarray or None
            Model SED fluxes in the target unit system, shape (n_filters,).
        """
        phases = np.asarray(self.sed_phases, dtype=float)
        n_filt = len(self.filters)

        original_enabled = {}
        for ds in list(bundle.datasets):
            try:
                enabled = bundle.get_value(f"{ds}@enabled@phoebe01")
                original_enabled[ds] = enabled
                if enabled:
                    bundle.disable_dataset(ds)
            except Exception:
                pass

        added_labels = []
        for filt in self.filters:
            label = ("sedlc_" + filt).replace(":", "_").replace("-", "_")
            if label in bundle.datasets:
                try:
                    bundle.remove_dataset("lc", dataset=label)
                except Exception:
                    try:
                        bundle.remove_dataset(dataset=label)
                    except Exception:
                        pass

            try:
                bundle.add_dataset(
                    "lc",
                    compute_phases=phases.tolist(),
                    passband=filt,
                    dataset=label,
                    pblum_mode="absolute",
                )
                added_labels.append(label)
            except Exception as e:
                logger.debug("Failed to add SED dataset for %s: %s", filt, e)

        if not added_labels:
            self._restore_datasets(bundle, original_enabled, added_labels)
            return None

        bundle.set_value("distance", value=dist, unit=u.pc)

        try:
            bundle.run_compute(compute="phoebe01")
        except Exception as e:
            logger.debug("PHOEBE SED compute failed: %s", e)
            self._restore_datasets(bundle, original_enabled, added_labels)
            return None

        fluxes_by_phase = np.full((len(phases), n_filt), np.nan, dtype=float)
        for i, filt in enumerate(self.filters):
            label = ("sedlc_" + filt).replace(":", "_").replace("-", "_")
            try:
                vals = bundle.get_value("fluxes", dataset=label, context="model")
                vals = np.asarray(vals, dtype=float).ravel()
                if vals.size == len(phases):
                    fluxes_by_phase[:, i] = vals
                elif vals.size == 1:
                    fluxes_by_phase[:, i] = vals[0]
            except Exception as e:
                logger.debug("Failed to extract fluxes for %s: %s", filt, e)

        all_finite = np.all(np.isfinite(fluxes_by_phase), axis=0)
        bandfluxes = np.full(n_filt, np.nan, dtype=float)
        bandfluxes[all_finite] = np.mean(fluxes_by_phase[:, all_finite], axis=0)

        self._restore_datasets(bundle, original_enabled, added_labels)

        bandfluxes = self._apply_extinction(bandfluxes, ebv)

        if not np.any(np.isfinite(bandfluxes) & (bandfluxes > 0)):
            return None

        return self.convert_units(bandfluxes)

    def _restore_datasets(self, bundle, original_enabled, added_labels):
        for label in added_labels:
            try:
                bundle.remove_dataset("lc", dataset=label)
            except Exception:
                try:
                    bundle.remove_dataset(dataset=label)
                except Exception:
                    pass
        for ds, enabled in original_enabled.items():
            if enabled:
                try:
                    bundle.enable_dataset(ds)
                except Exception:
                    pass

    def _apply_extinction(self, bandfluxes, ebv):
        if ebv is None or ebv == 0:
            return bandfluxes

        r_v = 3.1
        a_v = r_v * ebv
        result = bandfluxes.copy()

        for i, filt in enumerate(self.filters):
            if not np.isfinite(bandfluxes[i]) or bandfluxes[i] <= 0:
                continue
            pb = self._pb_cache[filt]
            wl_aa = np.asarray(pb.ptf_table["wl"], dtype=float) * 1e10
            R = np.asarray(pb.ptf_table["fl"], dtype=float)
            idx = np.argsort(wl_aa)
            wl_aa, R = wl_aa[idx], R[idx]
            R = np.clip(R, 0.0, None)

            den = np.trapz(R, wl_aa)
            if den <= 0:
                continue

            if np.any(wl_aa < 910) or np.any(wl_aa > 60000):
                logger.debug("Filter %s outside extinction valid range", filt)
                continue

            if self.extinction_law == "fitzpatrick99":
                A_lam = extinction.fitzpatrick99(wl_aa, a_v, r_v=r_v, unit="aa")
            elif self.extinction_law == "ccm89":
                A_lam = extinction.ccm89(wl_aa, a_v, r_v, unit="aa")
            else:
                raise ValueError(f"Unknown extinction law: {self.extinction_law}")

            att_lam = 10 ** (-0.4 * A_lam)
            att_eff = np.trapz(att_lam * R, wl_aa) / den
            att_eff = np.clip(att_eff, 1e-300, 1.0)
            result[i] *= att_eff

        return result

    def _bandflux_to_flam(self, bandflux_Wm2, filt):
        """Convert band-integrated flux (W/m^2) to f_lambda (erg/cm^2/s/A)."""
        i = self.filters.index(filt)
        bw = self._bandwidth_m[i]
        if not np.isfinite(bw) or bw <= 0:
            return np.nan
        flam_W_m2_m = bandflux_Wm2 / bw
        flam_cgs = flam_W_m2_m * 1e-7
        return flam_cgs

    def _bandflux_to_fnu(self, bandflux_Wm2, filt):
        """Convert band-integrated flux (W/m^2) to f_nu (Jy)."""
        i = self.filters.index(filt)
        bw = self._bandwidth_m[i]
        lam_piv = self._pivot_wl_m[i]
        if not np.isfinite(bw) or bw <= 0 or not np.isfinite(lam_piv):
            return np.nan
        flam_W_m2_m = bandflux_Wm2 / bw
        fnu_W_m2_Hz = flam_W_m2_m * (lam_piv ** 2) / C_LIGHT_M_S
        fnu_Jy = fnu_W_m2_Hz / 1e-26
        return fnu_Jy

    @staticmethod
    def _fnu_to_ABmag(fnu_Jy):
        """Convert f_nu (Jy) to AB magnitude."""
        if fnu_Jy <= 0 or not np.isfinite(fnu_Jy):
            return np.nan
        return -2.5 * np.log10(fnu_Jy / 3631.0)

    def _fnu_to_Vegamag(self, fnu_Jy, filt):
        """Convert f_nu (Jy) to Vega magnitude using passband zero points."""
        if fnu_Jy <= 0 or not np.isfinite(fnu_Jy):
            return np.nan
        zp = VEGA_ZP_JY.get(filt)
        if zp is None:
            logger.warning("No Vega zero point for %s; falling back to AB", filt)
            return self._fnu_to_ABmag(fnu_Jy)
        return -2.5 * np.log10(fnu_Jy / zp)

    def convert_units(self, bandfluxes_Wm2):
        """
        Convert array of band-integrated fluxes (W/m^2) to target units.

        Parameters
        ----------
        bandfluxes_Wm2 : ndarray, shape (n_filters,)

        Returns
        -------
        result : ndarray, shape (n_filters,)
        """
        result = np.full(len(self.filters), np.nan, dtype=float)
        for i, filt in enumerate(self.filters):
            bf = bandfluxes_Wm2[i]
            if not np.isfinite(bf) or bf <= 0:
                continue
            target = self.per_filter_units.get(filt, self.sed_units)
            if target == "flam":
                result[i] = self._bandflux_to_flam(bf, filt)
            elif target == "fnu_Jy":
                result[i] = self._bandflux_to_fnu(bf, filt)
            elif target == "ABmag":
                fnu = self._bandflux_to_fnu(bf, filt)
                result[i] = self._fnu_to_ABmag(fnu)
            elif target == "Vegamag":
                fnu = self._bandflux_to_fnu(bf, filt)
                result[i] = self._fnu_to_Vegamag(fnu, filt)
        return result
