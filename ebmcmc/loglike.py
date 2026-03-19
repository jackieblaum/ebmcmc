"""
Log-likelihood, prior, and forward model for eclipsing binary MCMC fitting.

Parameter vector layout
-----------------------
The MCMC sampler operates on an unconstrained parameter vector ``params``
whose length depends on the flags ``has_rv``, ``has_sed``, and ``ecc_bool``.
The flags ``has_rv`` and ``has_sed`` are independent — both may be True
simultaneously, enabling LC + SED + RV fitting.

Core block (always indices 0-6):
    0  u_q               logit(1 - q)  — mass ratio via sigmoid transform
    1  log_Msum           ln(M1 + M2) in solar masses
    2  log_teff1          ln(T_eff,1) in Kelvin
    3  log_tefffrac       ln(T_eff,2 / T_eff,1)
    4  log_rfrac          ln(R_equiv,2 / R_equiv,1)
    5  logit_rsumfrac     logit of (R1+R2)/a  (scaled by 1-eps)
    6  logit_cosi         logit(cos i)

SED block (if ``has_sed`` is True):
    7  log_dist            ln(distance) in parsec
    8  eta_alpha_sed       softplus^{-1}(alpha_sed) — SED fractional error

LC jitter (always present):
    +0  eta_sigma_lc       softplus^{-1}(sigma_lc) — LC additive jitter

RV block (if ``has_rv`` is True):
    +0  vgamma             systemic velocity (km/s)
    +1  eta_sigma_rv       softplus^{-1}(sigma_jit) — RV jitter

Optional eccentricity block (if ``ecc_bool`` is True):
    +0  ecc                eccentricity [0, 1)
    +1  per0_rad           argument of periastron (radians)

Final parameter (always last):
    psi_t0              unbounded phase offset; t0 = t0_ref + frac(psi_t0)*P
"""
import phoebe
import numpy as np
from binarysed.binarysed import SED
from ebmcmc.phoebe_sed import PhoebeSED
import time
import logging
from scipy.interpolate import interp1d
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

MODEL_TEMPLATE = None

_WORKER_STATE = {}

# --- Parameter vector index constants (core block, always present) ---
IDX_U_Q = 0
IDX_LOG_MSUM = 1
IDX_LOG_TEFF1 = 2
IDX_LOG_TEFFFRAC = 3
IDX_LOG_RFRAC = 4
IDX_LOGIT_RSUMFRAC = 5
IDX_LOGIT_COSI = 6

# --- Nuisance alpha / sigma bounds (shared between lnprior and lnlikelihood) ---
ALPHA_FLOOR = 0.03
ALPHA_CAP = 0.4

def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def softplus_inv(y):
    """Inverse of softplus: log(exp(y) - 1) for y > 0."""
    return np.log(np.expm1(y))

def soft_density_cap(x, upper=2.3, sigma=0.3):
    if x <= upper:
        return 0.0
    return -0.5 * ((x - upper) / sigma)**2

def frac(x):
    """Return fractional part in [0,1)."""
    return x - np.floor(x)

def von_mises_logpdf(phi, mu, kappa):
    """
    Unnormalized log von Mises density on the circle for phase in [0,1).
    mu, phi in [0,1), kappa >= 0. Returns kappa*cos(2π(φ-μ)) (drops constants).
    """
    return kappa * np.cos(2*np.pi*(phi - mu))

def log10_density_solar(mass_solar, radius_solar):
    """log10 density in units of rho_sun."""
    if (mass_solar <= 0) or (radius_solar <= 0):
        return np.inf
    return np.log10(mass_solar) - 3.0*np.log10(radius_solar)

def interp_periodic_phase(phi_model, y_model, phi_obs):
    """
    Linearly interpolate a periodic phase-folded curve onto observed phases.

    Parameters
    ----------
    phi_model : array_like
        Model phases (will be wrapped to [0, 1)).
    y_model : array_like
        Model values at ``phi_model``.
    phi_obs : array_like
        Observed phases at which to evaluate the interpolant.

    Returns
    -------
    y_out : ndarray or None
        Interpolated values at ``phi_obs``, or ``None`` if insufficient
        finite model points (< 4) or if the result contains non-finite values.
    """
    phi_model = np.asarray(phi_model, dtype=float)
    y_model   = np.asarray(y_model, dtype=float)
    phi_obs   = np.asarray(phi_obs, dtype=float)

    # finite mask
    m = np.isfinite(phi_model) & np.isfinite(y_model)
    if m.sum() < 4:
        return None

    phi = np.mod(phi_model[m], 1.0)
    y   = y_model[m]

    # sort
    s = np.argsort(phi)
    phi = phi[s]
    y   = y[s]

    # de-duplicate phases (keep first occurrence)
    # (or you could average duplicates; first-occurrence is fine for debug)
    phi_u, idx_u = np.unique(phi, return_index=True)
    y_u = y[idx_u]

    if phi_u.size < 4:
        return None

    # periodic wrap
    phi_per = np.concatenate([phi_u, phi_u[:1] + 1.0])
    y_per   = np.concatenate([y_u,   y_u[:1]])

    f = interp1d(phi_per, y_per, kind="linear", bounds_error=False, fill_value="extrapolate")
    y_out = f(np.mod(phi_obs, 1.0))

    if not np.all(np.isfinite(y_out)):
        return None
    return y_out



def _pool_init(data_dict, model_phases, use_ellc, sed_method="binarysed",
               sed_units="flam", per_filter_units=None, sed_phases=None,
               extinction_law="fitzpatrick99"):
    """Initializer for multiprocessing Pool: build a bundle once per process."""
    b, sed_obj, phoebe_sed_obj = _build_template_bundle(
        data_dict, model_phases, use_ellc, sed_method=sed_method,
        sed_units=sed_units, per_filter_units=per_filter_units,
        sed_phases=sed_phases, extinction_law=extinction_law,
    )
    _WORKER_STATE["b"] = b
    _WORKER_STATE["sed_obj"] = sed_obj
    _WORKER_STATE["phoebe_sed_obj"] = phoebe_sed_obj

def _build_template_bundle(data_dict, model_phases, use_ellc, sed_method="binarysed",
                           sed_units="flam", per_filter_units=None, sed_phases=None,
                           extinction_law="fitzpatrick99"):
    b = phoebe.default_binary()

    # ---- Decide how to choose compute_phases per dataset ----
    default_phases = phoebe.linspace(0, 0.9999, 201)

    # model_phases can be:
    # - None: use default_phases for all lc datasets
    # - array-like: use that for all lc datasets
    # - dict: per-dataset phases, with optional "default" fallback
    phases_map = None
    global_phases = None

    if model_phases is None:
        global_phases = default_phases
    elif isinstance(model_phases, dict):
        phases_map = model_phases
    else:
        global_phases = model_phases

    def _get_compute_phases(dataset_name: str):
        if phases_map is not None:
            # dataset-specific phases > "default" > default_phases
            return phases_map.get(dataset_name, phases_map.get("default", default_phases))
        return global_phases

    # ---- Add datasets once; do NOT pass fluxes (they’re the *data*). ----
    for dataset in data_dict:
        if dataset.startswith("lc"):
            compute_phases = _get_compute_phases(dataset)

            b.add_dataset(
                "lc",
                times=data_dict[dataset]["times"],
                sigmas=data_dict[dataset]["sigmas"],
                compute_phases=compute_phases,
                dataset=dataset,
            )

            dlow = dataset.lower()
            if "tess" in dlow:
                b.set_value(f"passband@{dataset}", value="TESS:T")
            elif "sdss" in dlow:
                b.set_value(f"passband@{dataset}", value="SDSS:g")
            elif "johnson" in dlow:
                b.set_value(f"passband@{dataset}", value="Johnson:V")
            elif "ztf" in dlow:
                b.set_value(f"passband@{dataset}", value="ZTF:r")
            elif "kepler" in dlow:
                b.set_value(f"passband@{dataset}", value="Kepler:mean")

        elif dataset.startswith("rv"):
            b.add_dataset(
                "rv",
                times={
                    "primary": data_dict[dataset]["primary_times"],
                    "secondary": data_dict[dataset]["secondary_times"],
                },
                rvs={
                    "primary": data_dict[dataset]["primary"],
                    "secondary": data_dict[dataset]["secondary"],
                },
                sigmas={
                    "primary": data_dict[dataset]["primary_sigmas"],
                    "secondary": data_dict[dataset]["secondary_sigmas"],
                },
                passband="Bolometric:900-40000",
                compute_phases=phoebe.linspace(-0.5,0.4999,100),
                dataset=dataset,  # optional but nice for consistency
            )

    b.set_value_all("ld_mode", "lookup" if use_ellc else "interp")
    b.set_value_all("ntriangles", value=1500)

    if use_ellc and 'ellcbackend' not in b.computes:
        b.add_compute('ellc', compute='ellcbackend')

    b.flip_constraint("mass@primary", solve_for="sma@binary@component")

    sed_obj = None
    phoebe_sed_obj = None

    if "sed" in data_dict:
        if sed_method == "binarysed":
            sed_obj = SED(data_dict["sed"])
        elif sed_method == "phoebe":
            phoebe_sed_obj = PhoebeSED(
                filters=data_dict["sed"]["filters"],
                sed_units=sed_units,
                per_filter_units=per_filter_units,
                sed_phases=sed_phases,
                extinction_law=extinction_law,
            )
        else:
            raise ValueError(f"Unknown sed_method: {sed_method}")

    return b, sed_obj, phoebe_sed_obj

def _t0_from_phase_param(params, period, t0_ref, has_rv, has_sed, ecc_bool):
    idx = 7
    if has_sed:
        idx += 2   # log_dist, eta_alpha_sed
    idx += 1       # eta_sigma_lc (always present)
    if has_rv:
        idx += 2   # vgamma, eta_sigma_rv
    if ecc_bool:
        idx += 2   # ecc, per0_rad
    psi_t0 = params[idx]
    phi0 = frac(psi_t0)
    t0 = t0_ref + phi0 * period
    return t0, phi0


def soft_barrier(x, lower=None, upper=None, k=10.0):
    """
    Smooth log-prior penalty that discourages values approaching or exceeding
    specified bounds. The penalty grows gradually near the boundary and
    increases roughly linearly once the bound is crossed.
    """
    pen = 0.0
    if lower is not None:
        # penalize when x < lower  -> z = (lower - x) > 0
        pen -= k * softplus(lower - x)
    if upper is not None:
        # penalize when x > upper  -> z = (x - upper) > 0
        pen -= k * softplus(x - upper)
    return pen


def roche_lobe_frac(q):
    """
    Roche lobe radii as fractions of the orbital separation (Eggleton 1983).

    Parameters
    ----------
    q : float
        Mass ratio M2/M1.

    Returns
    -------
    RL1, RL2 : float
        Roche lobe radius / semi-major axis for primary and secondary.
    """
    q = np.clip(q, 1e-6, 1e6)
    RL1 = 0.49*q**(-2/3) / (0.6*q**(-2/3) + np.log1p(q**(-1/3)))
    RL2 = 0.49*q**( 2/3) / (0.6*q**( 2/3) + np.log1p(q**( 1/3)))
    return RL1, RL2

def logit(p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p) - np.log1p(-p)


def lnprob(params, data_dict, u_q_init, asini_init, period_init, log_dist_init, t0_ref,
           log_Msum_init, teff1_init, C, ecc_bool, has_rv, has_sed, eclipsing, use_ellc,
           lc_coeff, rv_coeff, sed_coeff, model_phases, A_obs, sigma_A,
           prior_info=None, sed_method="binarysed"):
    start_time = time.time()
    if prior_info is None:
        prior_info = {}

    lp = lnprior(params, u_q_init, asini_init, period_init, log_dist_init, log_Msum_init, teff1_init, C,
                 ecc_bool, has_rv=has_rv, has_sed=has_sed, eclipsing=eclipsing,
                 A_obs=A_obs, sigma_A=sigma_A, prior_info=prior_info, t0_ref=t0_ref)
    if not np.isfinite(lp):
        return -np.inf

    ll = lnlikelihood(params, data_dict, C, period_init, t0_ref, ecc_bool, has_rv=has_rv, has_sed=has_sed,
                      use_ellc=use_ellc, lc_coeff=lc_coeff, rv_coeff=rv_coeff, sed_coeff=sed_coeff,
                      model_phases=model_phases, sed_method=sed_method)
    if not np.isfinite(ll):
        return -np.inf

    out = lp + ll
    if not np.isfinite(out):
        return -np.inf

    return out

def sigmoid(z): 
        return 1/(1+np.exp(-z))

def transform_params(params, period, has_rv, has_sed, ecc_bool, t0_ref=0.0):
    """
    Decode the unconstrained sampler vector into physical binary parameters.

    Parameters
    ----------
    params : array_like
        MCMC parameter vector (see module docstring for layout).
    period : float
        Orbital period in days (held fixed).
    has_rv : bool
        True if radial-velocity data is present.
    has_sed : bool
        True if SED data is present.
    ecc_bool : bool
        True if eccentricity is being fitted.
    t0_ref : float, optional
        Reference epoch for t0 computation (BJD).

    Returns
    -------
    tuple
        (q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist_or_None,
         vgamma_or_None, ecc_or_None, per0_rad_or_None)
    """
    u_q = params[0]
    log_Msum     = params[1]
    log_teff1    = params[2]
    log_tefffrac = params[3]
    log_rfrac    = params[4]
    logit_rsumfrac = params[5]
    logit_cosi   = params[6]

    idx = 7

    dist = None
    if has_sed:
        log_dist = params[idx]; dist = np.exp(log_dist); idx += 1
        eta_alpha_sed = params[idx]; idx += 1

    eta_sigma_lc = params[idx]; idx += 1  # always present

    vgamma = None
    if has_rv:
        vgamma = params[idx]; idx += 1
        eta_sigma_rv = params[idx]; idx += 1

    # ecc/per0
    if ecc_bool:
        ecc_val  = params[idx]
        per0_rad = params[idx + 1]
        idx += 2
    else:
        ecc_val  = None
        per0_rad = None

    psi_t0 = params[idx]
    idx += 1

    q = 1.0 - sigmoid(u_q)
    q = np.clip(q, 1e-6, 1.0 - 1e-6)
    Msum = np.exp(log_Msum)
    teff1 = np.exp(log_teff1)
    teff2 = np.exp(log_tefffrac + log_teff1)
    rfrac = np.exp(log_rfrac)
    eps = 1e-3
    rsumfrac = (1.0 - eps) * sigmoid(logit_rsumfrac)   # strictly < 1
    rfrac     = np.exp(log_rfrac)
    r1frac    = rsumfrac/(1+rfrac)
    r2frac    = rsumfrac - r1frac
    cosi = sigmoid(logit_cosi)
    incl = np.degrees(np.arccos(cosi))

    G = 2942.206217504419328179210424423218 # gravitational constant (solar units)
    a = (Msum * period**2 * G / (4 * np.pi**2))**(1/3)  # R_sun

    requiv1 = r1frac * a
    requiv2 = r2frac * a

    return q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad


def lnprior(params, u_q_init, asini_init, period, log_dist_init, log_Msum_init, teff1_init, C,
            ecc_bool, has_rv, has_sed, eclipsing, A_obs, sigma_A, prior_info=None, t0_ref=0.0):
    """
    Compute the log-prior for the binary model.

    Includes hard bounds on q, Msum, Teff, inclination, eccentricity,
    plus soft priors on Roche-lobe overflow, Teff1, distance (Gaia),
    mass-ratio (optional RV constraint), and SED fractional error.

    Parameters
    ----------
    params : array_like
        MCMC parameter vector.
    u_q_init : float
        Initial logit(1-q), used for scale reference.
    asini_init : float
        Initial a*sin(i) in R_sun.
    period : float
        Orbital period in days.
    log_dist_init : float or None
        Initial ln(distance) in parsec.
    log_Msum_init : float
        Initial ln(M1+M2).
    teff1_init : float
        Initial Teff of primary (K), for Gaussian prior center.
    C : float
        Unused legacy constant (kept for API compatibility).
    ecc_bool : bool
        Whether eccentricity is fitted.
    has_rv : bool
        Whether radial-velocity data is present.
    has_sed : bool
        Whether SED data is present.
    eclipsing : bool
        If True, apply sin(i) prior; if False, apply non-eclipse constraint.
    A_obs : float
        Observed ellipsoidal amplitude (for non-eclipsing systems).
    sigma_A : float
        Uncertainty on A_obs.
    prior_info : dict, optional
        Additional prior specifications (t0, gaia_dist, q_from_rv, etc.).
    t0_ref : float, optional
        Reference epoch (BJD).

    Returns
    -------
    float
        Log-prior value, or -inf if outside hard bounds.
    """
    if prior_info is None:
        prior_info = {}

    # === use transformed params with t0_ref ===
    tp = transform_params(params, period, has_rv=has_rv, has_sed=has_sed, ecc_bool=ecc_bool, t0_ref=t0_ref)
    if not isinstance(tp, tuple):
        return -np.inf
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad = tp

    t0, phi0 = _t0_from_phase_param(params, period, t0_ref=t0_ref,
                                    has_rv=has_rv, has_sed=has_sed, ecc_bool=ecc_bool)

    # Check priors
    if not (0 < q <= 1):
        return -np.inf
    if not (0.1 < Msum):
        return -np.inf
    if not (3500 < teff1 < 8500):
        return -np.inf
    if not (3500 < teff2 < 8500):
        return -np.inf

    if ecc_val is not None:
        if not (0.0 <= ecc_val < 1.0):
            return -np.inf

    if per0_rad is not None:
        if not (0.0 <= per0_rad < 2*np.pi):
            return -np.inf

    RL1, RL2 = roche_lobe_frac(q)

    # keep the stars within their RLs
    log_prior_roche1 = soft_barrier(requiv1/a, upper=RL1, k=2.0)
    log_prior_roche2 = soft_barrier(requiv2/a, upper=RL2, k=2.0)
    
    if not (0 < incl < 90):
        return -np.inf
    # NOTE: Alternative non-eclipse constraint using arccos(rsumfrac) was considered but replaced by soft_barrier

    M1 = Msum / (1.0 + q)
    M2 = q * M1

    logrho1 = log10_density_solar(M1, requiv1)
    logrho2 = log10_density_solar(M2, requiv2)

    # Broad sanity prior: only penalize absurdly high densities
    log_prior_density1 = soft_density_cap(logrho1, upper=2.3, k=3.0)
    log_prior_density2 = soft_density_cap(logrho2, upper=2.3, k=3.0)

    sigma_teff1 = 700
    log_prior_teff1 = -0.5 * ((teff1 - teff1_init) / sigma_teff1)**2

    r_sum_over_a = (requiv1 + requiv2) / a
    cosi = np.cos(np.radians(incl))

    # More priors for other parameters
    log_prior_noneclipse = 0
    log_prior_amplitude = 0
    if eclipsing:
        sin_i = np.sin(np.radians(incl))
        if sin_i <= 0 or not np.isfinite(sin_i):
            return -np.inf
        log_prior_incl = np.log(sin_i)  # sin(i) prior for random orientation
    else:
        log_prior_noneclipse = soft_barrier(cosi, lower=r_sum_over_a + 0.01, k=12.0)
        A_model = ( q*(requiv1/a)**3 + (requiv2/a)**3 ) * (np.sin(np.radians(incl))**2)

        # log-space weak match to observed semi-amplitude
        log_prior_amplitude = -0.5 * ((np.log(A_model + 1e-12) - np.log(A_obs)) / 0.5)**2

    log_prior_msum = 0

    log_prior_a = 0
    if has_sed:
        G     = 2942.2062175044193
        a_init = (np.exp(log_Msum_init) * period**2 * G / (4*np.pi**2))**(1/3)
        a_from_msum = (Msum * period**2 * G / (4*np.pi**2))**(1/3)
        log_prior_a = -0.5 * ((np.log(a_from_msum) - np.log(a_init))/0.7)**2

    log_prior_t0 = 0.0
    if "t0" in prior_info:
        t0_info = prior_info["t0"]
        if t0_info.get("flat", False):
            pass  # flat phase -> contributes 0
        else:
            phi_ref = float(t0_info.get("phi0_ref", 0.0))  # center in [0,1)
            kappa   = float(t0_info.get("kappa", 0.0))
            if kappa > 0:
                log_prior_t0 += von_mises_logpdf(phi0, phi_ref, kappa)

        # Optional absolute soft window on t0 (days)
        t0_lo = t0_info.get("t0_min", None)
        t0_hi = t0_info.get("t0_max", None)
        if (t0_lo is not None) or (t0_hi is not None):
            log_prior_t0 += soft_barrier(t0, lower=t0_lo, upper=t0_hi, k=8.0)

    # --- Distance prior: ONLY if has_sed ---
    if has_sed and ("gaia_dist" in prior_info):
        dist0   = prior_info["gaia_dist"]["dist0"]
        dist_lo = prior_info["gaia_dist"]["dist_lo"]
        dist_hi = prior_info["gaia_dist"]["dist_hi"]
        ruwe = prior_info["gaia_dist"]["ruwe"]

        sig_lo = np.log(dist0/dist_lo)
        sig_hi = np.log(dist_hi/dist0)

        scale = 1.0
        if ruwe is not None:
            if ruwe > 1.4:
                scale = 8.0
            elif ruwe > 1.2:
                scale = 2.0
        sig_lo *= scale
        sig_hi *= scale

        z = np.log(dist) - np.log(dist0)
        sigma = np.where(z < 0, sig_lo + 1e-12, sig_hi + 1e-12)
        log_prior_dist = -0.5 * (z / sigma)**2

        log_prior_dist_max = soft_barrier(dist, upper=1200.0, k=0.8)
    else:
        log_prior_dist = 0.0
        log_prior_dist_max = 0.0

    log_prior_alpha = 0
    if has_sed:
        eta_alpha_sed = params[8]  # log_dist at 7, eta_alpha_sed at 8 when has_sed
        tilde = 1/(1+np.exp(-eta_alpha_sed))                 # (0,1)
        alpha_sed = ALPHA_FLOOR + (ALPHA_CAP - ALPHA_FLOOR) * tilde

        # mild Beta prior on tilde (discourages living at the cap/floor)
        a, b = 2.0, 5.0
        log_prior_alpha = (a-1)*np.log(tilde + 1e-12) + (b-1)*np.log(1-tilde + 1e-12)

    if "q_from_rv" in prior_info:
        q0   = prior_info["q_from_rv"]["q0"]
        sigq = prior_info["q_from_rv"]["sigma_q"]
        # simple q-space Gaussian
        log_prior_q = -0.5 * ((q - q0) / (sigq + 1e-12))**2
    else:
        # Uniform prior on q in (0,1) implemented via Jacobian from u_q -> q
        u_q = params[0]
        s = sigmoid(u_q)  # s = 1 - q
        # log|dq/du| = log(s*(1-s))
        log_prior_q = np.log(s + 1e-12) + np.log1p(-s + 1e-12)

    asini = a * np.sin(np.radians(incl))
    log_prior_asini = 0
    if "asini_from_rv" in prior_info:
        asini0     = prior_info["asini_from_rv"]["asini0"]
        frac_sigma = prior_info["asini_from_rv"].get("frac_sigma", 0.03)
        sig_asini  = frac_sigma * asini0
        log_prior_asini = -0.5 * ((asini - asini0) / (sig_asini + 1e-12))**2

    if "msum_cap" in prior_info:
        mcap = prior_info["msum_cap"]
        log_prior_msum_upper = soft_barrier(Msum, upper=mcap, k=3.0)
    else:
        # optional soft safety net instead of a hard cap
        log_prior_msum_upper = soft_barrier(Msum, upper=15.0, k=3.0)

    log_prior_total = (log_prior_q + log_prior_asini + log_prior_roche1 + log_prior_roche2 +
                        log_prior_density1 + log_prior_density2 +
                       log_prior_a + log_prior_msum + log_prior_msum_upper + log_prior_dist +
                       log_prior_dist_max + log_prior_amplitude + log_prior_noneclipse +
                       log_prior_t0 + log_prior_alpha + log_prior_teff1)

    if eclipsing:
        log_prior_total += log_prior_incl

    return log_prior_total

def forward_model(params, data_dict, C, period, t0, ecc_bool, has_rv, has_sed, use_ellc,
                  model_phases=None, sed_method="binarysed"):
    """
    Run PHOEBE (or ellc) forward model and return predicted observables.

    Parameters
    ----------
    params : array_like
        MCMC parameter vector.
    data_dict : dict
        Observed data keyed by dataset name.
    C : float
        Unused legacy constant.
    period : float
        Orbital period in days.
    t0 : float
        Time of superior conjunction (BJD).
    ecc_bool : bool
        Whether eccentricity is fitted.
    has_rv : bool
        Whether radial-velocity data is present.
    has_sed : bool
        Whether SED data is present.
    use_ellc : bool
        If True, use the ellc backend instead of PHOEBE.
    model_phases : dict or array_like or None
        Compute phases for light-curve evaluation.

    Returns
    -------
    tuple
        (y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model)
        where any element may be None if not applicable. Returns
        (None, None, None, None) on failure.
    """
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad = transform_params(params, period, has_rv=has_rv, has_sed=has_sed, ecc_bool=ecc_bool)

    if not _WORKER_STATE:  # fallback for single-process/no-pool runs
        _pool_init(data_dict, model_phases, use_ellc, sed_method=sed_method)

    base_b = _WORKER_STATE["b"]
    try:
        b = base_b.copy()        # preferred if available
    except Exception:
        b = base_b.deepcopy()

    b.set_value_all("ld_mode", "lookup" if use_ellc else "interp")
    b.set_value_all("ntriangles", value=1500)
    b.set_value("eclipse_method", value="native")
    b.set_value_all("pblum_mode", "component-coupled")

    # set system parameters
    M1 = Msum/(1.0+q)
    b.set_value("q@binary@component", q)
    b.set_value("mass@primary@component", M1)
    b.set_value("period@binary@component", period)
    b.set_value("teff@primary@component", teff1)
    b.set_value("teff@secondary@component", teff2)
    b.set_value("requiv@primary@component", requiv1)
    b.set_value("requiv@secondary@component", requiv2)
    b.set_value("incl@binary@component", incl)
    b.set_value("t0_supconj@binary@component", t0)

    if has_rv:
        b.set_value("vgamma@system", vgamma)

    if ecc_bool:
        # set your ecc/per0 when you wire them
        b.set_value("ecc@binary@component", ecc_val)
        per0_deg = np.rad2deg(per0_rad)
        b.set_value("per0@binary@component", per0_deg)

    # TODO: Re-enable gravity darkening and per-star LD coefficients for hot stars (T>8000K)

    # compute
    if use_ellc:
        b.run_compute(compute='ellcbackend')
    else:
        b.run_compute(compute='phoebe01')

    # collect model, normalize via median to the data
    y_pred_lc = [] 
    y_pred_rv_primary = None 
    y_pred_rv_secondary = None 
    for dataset in b.datasets: 
        if dataset.startswith("lc"): 
            # Get model fluxes and times 
            model_fluxes = b.get_value(f"fluxes@model@{dataset}") 
            model_times = b.get_value(f"times@model@{dataset}") 
            model_phases = b.to_phase(model_times)
            observed_times = b.get_value(f"times@{dataset}@dataset") 
            observed_phases = b.to_phase(observed_times) 
            # Ensure model_phases and model_fluxes are sorted 
            sort_idx = np.argsort(model_phases)
            phi = np.array(model_phases)[sort_idx]
            flux = np.array(model_fluxes)[sort_idx]

            # append periodic endpoint (phi+1 -> wrap)
            phi_per  = np.concatenate([phi,  phi[:1] + 1.0])
            flux_per = np.concatenate([flux, flux[:1]])

            interp_func = interp1d(phi_per, flux_per, kind="linear",  # linear is safer at wrap
                                bounds_error=False, fill_value="extrapolate")

            observed_phases = b.to_phase(observed_times)
            interpolated_fluxes = interp_func(observed_phases) 
            y_data = data_dict[dataset]["data"] 
            m = np.isfinite(y_data) & np.isfinite(interpolated_fluxes)
            scale = np.nanmedian(y_data[m]) / np.nanmedian(interpolated_fluxes[m])
            interpolated_fluxes *= scale
            y_pred_lc.append(interpolated_fluxes) 

        elif dataset.startswith("rv"):

            # --- model phases and model RVs (already in phase space) ---
            model_times = b.get_value(f"times@primary@model@{dataset}") 
            model_phases = b.to_phase(model_times)

            rv_model_primary   = b.get_value(f"rvs@model@{dataset}@primary")
            rv_model_secondary = b.get_value(f"rvs@model@{dataset}@secondary")

            # --- observed phases (from observed times OR already stored) ---
            # Prefer: store observed phases in data_dict once and reuse.
            obs_times_primary   = data_dict[dataset]["primary_times"]
            obs_times_secondary = data_dict[dataset]["secondary_times"]

            obs_ph_primary = b.to_phase(obs_times_primary)
            obs_ph_secondary = b.to_phase(obs_times_secondary)

            # sanity: if PHOEBE is giving NaNs, catch before interp
            for name, arr in [("model_phases", model_phases),
                            ("rv_model_primary", rv_model_primary),
                            ("rv_model_secondary", rv_model_secondary)]:
                arr = np.asarray(arr, dtype=float)
                if np.any(~np.isfinite(arr)):
                    return None, None, None, None

            y1 = interp_periodic_phase(model_phases, rv_model_primary,   obs_ph_primary)
            y2 = interp_periodic_phase(model_phases, rv_model_secondary, obs_ph_secondary)

            if (y1 is None) or (y2 is None):
                return None, None, None, None

            y_pred_rv_primary   = y1
            y_pred_rv_secondary = y2
                
    sed_model = None
    if has_sed and ("sed" in data_dict):
        if np.isnan(data_dict["sed"]["dist"]):
            logger.warning("No distance available; leaving SED out of the fit.")
            sed_model = None
        else:
            try:
                if sed_method == "phoebe":
                    phoebe_sed_obj = _WORKER_STATE.get("phoebe_sed_obj")
                    if phoebe_sed_obj is None:
                        raise RuntimeError(
                            "sed_method='phoebe' but PhoebeSED was not initialized. "
                            "Ensure _pool_init is called with sed_method='phoebe'."
                        )
                    ebv = data_dict["sed"].get("ebv", 0.0)
                    compute = "ellcbackend" if use_ellc else "phoebe01"
                    sed_model = phoebe_sed_obj.compute_sed(b, dist, ebv=ebv, compute=compute)
                else:
                    sed_obj = _WORKER_STATE["sed_obj"]
                    wavelengths = data_dict["sed"]["wavelengths"]
                    logg1 = b.get_value("logg@primary@component")
                    logg2 = b.get_value("logg@secondary@component")
                    sed_model = sed_obj.create_apparent_sed(
                        wavelengths, teff1, teff2, requiv1, requiv2,
                        logg1, logg2, dist, select_wavelengths=True,
                    )

                if sed_model is not None and not np.all(np.isfinite(sed_model)):
                    raise ValueError("SED contains non-finite values.")
            except Exception as e:
                logger.debug("[SED ERROR] %s  dist=%.3f, teff1=%.1f, teff2=%.1f, "
                             "r1=%.4f, r2=%.4f, incl=%.3f",
                             e, dist, teff1, teff2, requiv1, requiv2, incl)
                return None, None, None, None
    else:
        sed_model = None 
            
    return y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model 

def lnlikelihood(params, data_dict, C, period, t0_ref, ecc_bool, has_rv, has_sed, use_ellc, model_phases,
                 lc_coeff=1, rv_coeff=1, sed_coeff=1, sed_method="binarysed"):
    """
    Compute the log-likelihood for the binary model.

    Evaluates the forward model against observed light curves, radial
    velocities, and/or SED data. Includes learned jitter terms and
    inverse-variance weighting across data types.

    Parameters
    ----------
    params : array_like
        MCMC parameter vector.
    data_dict : dict
        Observed data keyed by dataset name.
    C : float
        Unused legacy constant.
    period : float
        Orbital period in days.
    t0_ref : float
        Reference epoch (BJD).
    ecc_bool : bool
        Whether eccentricity is fitted.
    has_rv : bool
        Whether radial-velocity data is present.
    has_sed : bool
        Whether SED data is present.
    use_ellc : bool
        If True, use the ellc backend.
    model_phases : dict or array_like or None
        Compute phases for light-curve evaluation.
    lc_coeff, rv_coeff, sed_coeff : float
        Unused weighting coefficients (reserved for future use).

    Returns
    -------
    float
        Log-likelihood value, or -inf on failure.
    """
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad = transform_params(
        params, period, has_rv=has_rv, has_sed=has_sed, ecc_bool=ecc_bool, t0_ref=t0_ref
    )

    t0, phi0 = _t0_from_phase_param(params, period, t0_ref=t0_ref,
                                    has_rv=has_rv, has_sed=has_sed, ecc_bool=ecc_bool)

    try:
        y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model = forward_model(
            params, data_dict, C, period, t0, ecc_bool, has_rv=has_rv, has_sed=has_sed,
            use_ellc=use_ellc, model_phases=model_phases, sed_method=sed_method,
        )
        if y_pred_lc is None:
            return -np.inf
    except ValueError as e:
        logger.debug("Forward model exception: %s", e)
        return -np.inf

    # Noise / nuisance params — walk indices dynamically
    idx = 7
    eta_alpha_sed = None
    if has_sed:
        # log_dist at idx, eta_alpha_sed at idx+1
        eta_alpha_sed = params[idx + 1]
        idx += 2
    eta_sigma_lc = params[idx]  # always present
    idx += 1
    eta_sigma_rv = None
    if has_rv:
        # vgamma at idx, eta_sigma_rv at idx+1
        eta_sigma_rv = params[idx + 1]
        
    lc_datasets = []
    rv_dataset = None
    for dataset in data_dict:
        if dataset.startswith('lc'):
            lc_datasets.append(dataset)
        elif dataset.startswith('rv'):
            rv_dataset = dataset
    
    # Calculate chi-squared for light curves
    chi2_lc = 0
    N_lc_points = 0
    lc_params = 7 # q, r1, r2, i, teffratio
    if ecc_bool:
        lc_params += 2 # ecc, per0
    for dataset, y_pred in zip(lc_datasets, y_pred_lc):
        data_lc = data_dict[dataset]["data"]
        sigma_lc_obs = data_dict[dataset]["sigmas"]
        
        # map to physical jitters with floors
        jitter = softplus(eta_sigma_lc)    # additive (relative flux)
        var = sigma_lc_obs**2 + jitter**2
        chi2_lc += np.sum((data_lc - y_pred) ** 2 / var) + np.sum(np.log(2*np.pi*var))
        N_lc_points += len(data_lc)

    # Calculate chi-squared for RVs, if present
    # Calculate chi-squared for RVs, if present
    chi2_rv = 0.0
    N_rv_points = 0

    rv_params = 5  # m1, m2, i, a, vgamma
    if ecc_bool:
        rv_params += 3  # ecc, per0, t0_supconj

    if (
        y_pred_rv_primary is not None
        and y_pred_rv_secondary is not None
        and rv_dataset is not None
    ):
        sigma_rv1 = np.asarray(data_dict[rv_dataset]["primary_sigmas"], dtype=float)
        sigma_rv2 = np.asarray(data_dict[rv_dataset]["secondary_sigmas"], dtype=float)
        data_rv1  = np.asarray(data_dict[rv_dataset]["primary"], dtype=float)
        data_rv2  = np.asarray(data_dict[rv_dataset]["secondary"], dtype=float)

        # NEW: RV jitter (km/s) added in quadrature
        if eta_sigma_rv is not None:
            sigma_jit = softplus(eta_sigma_rv)      # >0
            var1 = sigma_rv1**2 + sigma_jit**2
            var2 = sigma_rv2**2 + sigma_jit**2
        else:
            var1 = sigma_rv1**2
            var2 = sigma_rv2**2

        if rv_dataset is not None:
            for name, arr in [("pred1", y_pred_rv_primary), ("pred2", y_pred_rv_secondary),
                            ("sig1", data_dict[rv_dataset]["primary_sigmas"]),
                            ("sig2", data_dict[rv_dataset]["secondary_sigmas"]),
                            ("dat1", data_dict[rv_dataset]["primary"]),
                            ("dat2", data_dict[rv_dataset]["secondary"])]:
                arr = np.asarray(arr, dtype=float)
                if np.any(~np.isfinite(arr)):
                    logger.debug("[RV BAD] %s first bad idx %d", name, np.where(~np.isfinite(arr))[0][0])
                    return -np.inf

        # log-likelihood for each assignment, including normalization
        logL0 = (
            -0.5 * np.sum((data_rv1 - y_pred_rv_primary )**2 / var1 + np.log(2*np.pi*var1))
            -0.5 * np.sum((data_rv2 - y_pred_rv_secondary)**2 / var2 + np.log(2*np.pi*var2))
        )

        logL1 = (
            -0.5 * np.sum((data_rv1 - y_pred_rv_secondary)**2 / var1 + np.log(2*np.pi*var1))
            -0.5 * np.sum((data_rv2 - y_pred_rv_primary )**2 / var2 + np.log(2*np.pi*var2))
        )

        # marginalize over swap (equal prior)
        logL_rv = logsumexp([logL0, logL1]) - np.log(2.0)

        # store as effective chi2 (optional, for your weighting scheme)
        chi2_rv = -2.0 * logL_rv

        N_rv_points = len(data_rv1) + len(data_rv2)


    # Calculate chi-squared for SED, if present
    chi2_sed = 0.0
    N_sed_points = 0

    if has_sed and ("sed" in data_dict) and (sed_model is not None):
        data_sed = data_dict["sed"]["fluxes"]
        sigma_sed = data_dict["sed"]["flux_errs"]
        alpha_sed   = ALPHA_FLOOR + (ALPHA_CAP - ALPHA_FLOOR) * (1.0 / (1.0 + np.exp(-eta_alpha_sed)))
        var = sigma_sed**2 + (alpha_sed * sed_model)**2
        chi2_sed = np.sum((data_sed - sed_model) ** 2 / var) + np.sum(np.log(2*np.pi*var))
        N_sed_points = len(data_sed)

    # Calculate weights based on number of points
    w_LC = 0
    w_RV = 0
    w_SED = 0
    if N_lc_points != 0:
        w_LC = (N_rv_points + N_sed_points) / N_lc_points if (N_rv_points + N_sed_points) > 0 else 1
    if N_rv_points != 0:
        w_RV = (N_lc_points + N_sed_points) / N_rv_points if (N_lc_points + N_sed_points) > 0 else 1
    if N_sed_points != 0:
        w_SED = (N_lc_points + N_rv_points) / N_sed_points if (N_lc_points + N_rv_points) > 0 else 1

    chi2 = w_LC * chi2_lc + w_RV * chi2_rv + w_SED * chi2_sed
    out = -0.5 * chi2
    if not np.isfinite(out):
        logger.debug("[NAN LIKELIHOOD] chi2_lc=%s chi2_rv=%s chi2_sed=%s", chi2_lc, chi2_rv, chi2_sed)
        return -np.inf
    return out