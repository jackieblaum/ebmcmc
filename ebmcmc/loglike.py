import phoebe
import numpy as np
from binarysed.binarysed import SED
import sys
import time
import logging
from scipy.interpolate import interp1d

MODEL_TEMPLATE = None

_WORKER_STATE = {}

def frac(x):
    """Return fractional part in [0,1)."""
    return x - np.floor(x)

def von_mises_logpdf(phi, mu, kappa):
    """
    Unnormalized log von Mises density on the circle for phase in [0,1).
    mu, phi in [0,1), kappa >= 0. Returns kappa*cos(2π(φ-μ)) (drops constants).
    """
    return kappa * np.cos(2*np.pi*(phi - mu))


def _pool_init(data_dict, model_phases, use_ellc):
    """Initializer for multiprocessing Pool: build a bundle once per process."""
    b, sed_obj = _build_template_bundle(data_dict, model_phases, use_ellc)
    _WORKER_STATE["b"] = b
    _WORKER_STATE["sed_obj"] = sed_obj

def _build_template_bundle(data_dict, model_phases, use_ellc):
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
                dataset=dataset,  # optional but nice for consistency
            )

    b.set_value_all("ld_mode", "lookup" if use_ellc else "interp")
    b.set_value_all("ntriangles", value=1500)

    if use_ellc and 'ellcbackend' not in b.computes:
        b.add_compute('ellc', compute='ellcbackend')

    b.flip_constraint("mass@primary", solve_for="sma@binary@component")

    if "sed" in data_dict:
        sed_obj = SED(data_dict["sed"])

        return b, sed_obj

    return b, None

def soft_barrier(x, lower=None, upper=None, k=10.0):
    """
    Returns a non-positive penalty; 0 if inside [lower, upper].
    Penalty grows smoothly (≈k*softplus) when outside.
    """
    pen = 0.0
    if lower is not None:
        pen -= k * np.log1p(np.exp(lower - x))
    if upper is not None:
        pen -= k * np.log1p(np.exp(x - upper))
    return pen

def roche_lobe_frac(q):
    # Eggleton 1983; RL/a for primary (1) and secondary (2)
    q = np.clip(q, 1e-6, 1e6)
    RL1 = 0.49*q**(-2/3) / (0.6*q**(-2/3) + np.log1p(q**(-1/3)))
    RL2 = 0.49*q**( 2/3) / (0.6*q**( 2/3) + np.log1p(q**( 1/3)))
    return RL1, RL2

def logit(p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return np.log(p) - np.log1p(1-p)


def lnprob(params, data_dict, logit_q_init, asini_init, period_init, log_dist_init, t0_ref,
           log_Msum_init, C, ecc_bool, rv_bool, eclipsing, use_ellc,
           lc_coeff, rv_coeff, sed_coeff, model_phases, A_obs, sigma_A, prior_info=None):
    start_time = time.time()
    if prior_info is None:
        prior_info = {}

    lp = lnprior(params, logit_q_init, asini_init, period_init, log_dist_init, log_Msum_init, C,
                 ecc_bool, rv_bool, eclipsing, A_obs, sigma_A, prior_info, t0_ref=t0_ref)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlikelihood(params, data_dict, C, period_init, t0_ref, ecc_bool, rv_bool, use_ellc,
                             lc_coeff=lc_coeff, rv_coeff=rv_coeff, sed_coeff=sed_coeff,
                             model_phases=model_phases)


def sigmoid(z): 
        return 1/(1+np.exp(-z))

def transform_params(params, period, rv_bool, ecc_bool, t0_ref=0.0):
    """
    Decode the sampler vector -> physical params. Also maps psi_t0 -> phi0 -> t0.
    Returns: q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, t0, phi0
    """
    logit_q      = params[0]
    log_Msum     = params[1]
    log_teff1    = params[2]
    log_tefffrac = params[3]
    log_rfrac    = params[4]
    logit_rsumfrac = params[5]
    logit_cosi   = params[6]
    log_dist     = params[7]

    # Then branch exactly like initial_guess
    if rv_bool:
        vgamma = params[8]
        idx = 9
    else:
        alpha_sed = params[8]
        sigma_lc  = params[9]
        idx = 10

    # Ecc/per0 placement depends on BOTH ecc and whether RVs are included
    if ecc_bool and rv_bool:
        ecc_val = params[idx]       # matches initial_guess[9]
        per0_rad = params[idx + 1]  # matches initial_guess[10]
        idx += 2
    elif ecc_bool:
        ecc_val = params[idx]       # matches initial_guess[10]
        per0_rad = params[idx + 1]  # matches initial_guess[11]
        idx += 2

    q = sigmoid(logit_q)
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
    dist = np.exp(log_dist)

    G = 2942.206217504419328179210424423218 # gravitational constant (solar units)
    a = (Msum * period**2 * G / (4 * np.pi**2))**(1/3)  # R_sun

    requiv1 = r1frac * a
    requiv2 = r2frac * a

    # phi0 = frac(psi_t0)       # [0,1)
    # t0   = t0_ref + phi0*period

    if rv_bool and ecc_bool:
        return q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad
    elif rv_bool: 
        return q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, None, None
    else:
        return q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, None, None, None 


def lnprior(params, logit_q_init, asini_init, period, log_dist_init, log_Msum_init, C,
            ecc_bool, rv_bool, eclipsing, A_obs, sigma_A, prior_info=None, t0_ref=0.0):
    if prior_info is None:
        prior_info = {}

    # === use transformed params with t0_ref ===
    tp = transform_params(params, period, rv_bool, ecc_bool, t0_ref=t0_ref)
    if not isinstance(tp, tuple):
        return -np.inf
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad = tp
    # Check priors
    if not (0 < q <= 1):
        # print(f"q value: {q}")
        return -np.inf
    if not (0.1 < Msum):  # Stellar mass range
        # print(f"Msum value: {Msum}")
        return -np.inf
    # if not (period_init-0.1 < period < period_init+0.1):
    #     print(f"period value: {period}")
    #     return -np.inf
    # if not (t0_init-0.1 < t0_supconj < t0_init+0.1):
    #     print(f"t0_supconj value: {t0_supconj}")
    #     return -np.inf
    if not (2500 < teff1 < 50000):
        # print(f"teff_secondary value: {teff1}")
        return -np.inf
    if not (2500 < teff2 < 50000):
        # print(f"teff_secondary value: {teff2}")
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
        # print(f"incl value: {incl}")
        return -np.inf
    # else:
    #     i_max_rad = np.arccos(requivsumfrac)
    #     i_max = np.degrees(i_max_rad)
    #     if not (0 < incl < i_max):
    #         print(f"requivsumfrac: {requivsumfrac}")
    #         print(f"i_max: {i_max}")
    #         print(f"incl value: {incl}")
    #         return -np.inf
    # if not np.all((pblums > 0) & (pblums < 1e6)):
    #     return -np.inf

    # More priors for other parameters
    if eclipsing:
        log_prior_incl = np.log(np.sin(np.radians(incl)))  # sin(i) prior for random orientation

    # Compute C using your MAP estimates
    # C = log_Msum_init - 3 * log_requiv1_init

    # Compute the ridge prior in log space
    # delta = log_Msum - 3 * log_requiv1 - C
    # sigma_ridge = 0.02  # Adjust this width as needed (this is in log space now)
    # log_prior_ridge = -0.5 * (delta / sigma_ridge) ** 2

    # log_prior_period = -0.5 * ((period - period_init) / (0.001 * period_init))**2
    # log_prior_t0_supconj = -0.5 * ((t0_supconj - t0_init) / (0.02 * period_init))**2
    # log_prior_K1 = -0.5 * ((K1 - K1_init) / (0.05 * K1_init))**2
    # log_prior_K2 = -0.5 * ((K2 - K2_init) / (0.05 * K2_init))**2

    r_sum_over_a = (requiv1 + requiv2) / a
    cosi = np.cos(np.radians(incl))

    # we want: cos(i) > r_sum_over_a  (i.e. below eclipse limit)
    # add a soft penalty if we violate it
    log_prior_noneclipse = soft_barrier(cosi, lower=r_sum_over_a + 0.01, k=12.0)

    alpha_IMF = 2.3
    log_prior_msum = -alpha_IMF * np.log(Msum)

    G     = 2942.2062175044193
    a_init = (np.exp(log_Msum_init) * period**2 * G / (4*np.pi**2))**(1/3)
    a_from_msum = (Msum * period**2 * G / (4*np.pi**2))**(1/3)
    log_prior_a = -0.5 * ((np.log(a_from_msum) - np.log(a_init))/0.7)**2

    A_model = ( q*(requiv1/a)**3 + (requiv2/a)**3 ) * (np.sin(np.radians(incl))**2)

    # log-space weak match to observed semi-amplitude
    log_prior_amplitude = -0.5 * ((np.log(A_model + 1e-12) - np.log(A_obs)) / 0.5)**2

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

    if "gaia_dist" in prior_info:
        dist0   = prior_info["gaia_dist"]["dist0"]
        dist_lo = prior_info["gaia_dist"]["dist_lo"]
        dist_hi = prior_info["gaia_dist"]["dist_hi"]
        ruwe = prior_info["gaia_dist"]["ruwe"]

        sig_lo = np.log(dist0/dist_lo)  # ≈ log-space σ below median
        sig_hi = np.log(dist_hi/dist0)  # ≈ log-space σ above median

        scale = 1.0
        if ruwe is not None:
            if ruwe > 1.4:
                scale = 8.0  
            elif ruwe > 1.2:
                scale = 2.0   # borderline → just loosen a bit

        sig_lo *= scale
        sig_hi *= scale

        z = np.log(dist) - np.log(dist0)
        sigma = np.where(z < 0, sig_lo + 1e-12, sig_hi + 1e-12)
        log_prior_dist = -0.5 * (z / sigma)**2
    else:
        log_prior_dist = 0.0
    log_prior_dist_max = soft_barrier(dist, upper=1200.0, k=0.8)

    alpha_floor = 0.03
    alpha_cap   = 0.4

    eta_alpha_sed = params[8]
    tilde = 1/(1+np.exp(-eta_alpha_sed))                 # (0,1)
    alpha_sed = alpha_floor + (alpha_cap - alpha_floor) * tilde

    # mild Beta prior on tilde (discourages living at the cap/floor)
    a, b = 2.0, 5.0
    log_prior_alpha = (a-1)*np.log(tilde + 1e-12) + (b-1)*np.log(1-tilde + 1e-12)

    if "q_from_rv" in prior_info:
        q0   = prior_info["q_from_rv"]["q0"]
        sigq = prior_info["q_from_rv"]["sigma_q"]
        # simple q-space Gaussian
        log_prior_q = -0.5 * ((q - q0) / (sigq + 1e-12))**2
    else:
        # Uniform priors return 0 (log(1)); if Gaussian, use -0.5 * ((param - mu)/sigma)**2
        q_init = sigmoid(logit_q_init)
        sigma_logit_q = 0.35    # ~weak; tune 0.2–0.6 if needed
        log_prior_q = -0.5 * ((logit(q) - logit(q_init)) / sigma_logit_q)**2

    asini = a * np.sin(np.radians(incl))
    if "asini_from_rv" in prior_info:
        asini0     = prior_info["asini_from_rv"]["asini0"]
        frac_sigma = prior_info["asini_from_rv"].get("frac_sigma", 0.03)
        sig_asini  = frac_sigma * asini0
        log_prior_asini = -0.5 * ((asini - asini0) / (sig_asini + 1e-12))**2
    else:
        # Gaussian prior on asini   
        log_prior_asini = -0.5 * ((asini - asini_init) / (0.03*asini_init))**2

    if "msum_cap" in prior_info:
        mcap = prior_info["msum_cap"]
        log_prior_msum_upper = soft_barrier(Msum, upper=mcap, k=3.0)
    else:
        # optional soft safety net instead of a hard cap
        log_prior_msum_upper = soft_barrier(Msum, upper=15.0, k=3.0)

    log_prior_total = (log_prior_q + log_prior_asini + log_prior_roche1 + log_prior_roche2 +
                       log_prior_a + log_prior_msum + log_prior_msum_upper + log_prior_dist +
                       log_prior_dist_max + log_prior_amplitude + log_prior_noneclipse +
                       log_prior_t0 + log_prior_alpha)

    if eclipsing:
        log_prior_total += log_prior_incl

    return log_prior_total

def forward_model(params, data_dict, C, period, t0, ecc_bool, rv_bool, use_ellc, model_phases=None):
    # Decode params (unchanged)
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad = transform_params(params, period, rv_bool, ecc_bool)

    if not _WORKER_STATE:  # fallback for single-process/no-pool runs
        _pool_init(data_dict, model_phases, use_ellc)

    base_b = _WORKER_STATE["b"]
    sed_obj = _WORKER_STATE["sed_obj"]
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

    if rv_bool:
        b.set_value("vgamma@system", vgamma)

    if ecc_bool:
        # set your ecc/per0 when you wire them
        b.set_value("ecc@binary@component", ecc_val)
        per0_deg = np.rad2deg(per0_rad)
        b.set_value("per0@binary@component", per0_deg)

    b.set_value("gravb_bol@primary",   value=(0.9  if teff1 > 8000 else 0.32))
    b.set_value("irrad_frac_refl_bol@primary",   value=(1.0  if teff1 > 8000 else 0.6))
    b.set_value("gravb_bol@secondary", value=(0.9  if teff2 > 8000 else 0.32))
    b.set_value("irrad_frac_refl_bol@secondary", value=(1.0  if teff2 > 8000 else 0.6))

    # limb-darkening source: set *both* branches explicitly
    logg_primary  = b.get_value("logg@primary@component")
    logg_secondary= b.get_value("logg@secondary@component")

    src1 = 'phoenix' if (teff1 < 3500 or logg_primary  > 5) else 'ck2004'
    src2 = 'phoenix' if (teff2 < 3500 or logg_secondary> 5) else 'ck2004'
    b.set_value_all('ld_coeffs_source_bol@primary',  value=src1)
    b.set_value_all('ld_coeffs_source_bol@secondary',value=src2)

    # compute
    if use_ellc:
        b.run_compute(compute='ellcbackend')
    else:
        b.run_compute(compute='phoebe01')

    # collect model, normalize via median to the data (Option A)
    y_pred_lc = [] 
    y_pred_rv_primary = None 
    y_pred_rv_secondary = None 
    for dataset in b.datasets: 
        if dataset.startswith("lc"): 
            # Get model fluxes and times 
            model_fluxes = b.get_value(f"fluxes@model@{dataset}") 
            model_times = b.get_value(f"times@model@{dataset}") 
            model_phases = b.to_phase(model_times)
            # Convert to phase model_phases = b.to_phase(model_times) 
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
            # if you have an out-of-eclipse mask, use it; else median is ok for ellipsoidal LCs 
            m = np.isfinite(y_data) & np.isfinite(interpolated_fluxes) 
            scale = np.nanmedian(y_data[m]) / np.nanmedian(interpolated_fluxes[m]) 
            interpolated_fluxes *= scale 
            # Unsort back to original order of observed phases/times 
            # inverse_idx = np.argsort(sort_idx) 
            # unsorted_interpolated_fluxes = interpolated_fluxes[inverse_idx] 
            y_pred_lc.append(interpolated_fluxes) 

        elif dataset.startswith("rv"): 
            y_pred_rv_primary = b.get_value(f"rvs@model@{dataset}@primary") 
            y_pred_rv_secondary = b.get_value(f"rvs@model@{dataset}@secondary") 
            
    if "sed" in data_dict: 
        if np.isnan(data_dict["sed"]["dist"]): 
            print("No distance available; leaving SED out of the fit.") 
            sed_model = None 
        else: 
            try:
                wavelengths = data_dict["sed"]["wavelengths"] 
                logg1 = b.get_value("logg@primary@component") 
                logg2 = b.get_value("logg@secondary@component") 
                sed_model = sed_obj.create_apparent_sed(wavelengths, teff1, teff2, requiv1, requiv2, logg1, logg2, dist, select_wavelengths=True ) 
                
                if not np.all(np.isfinite(sed_model)):
                    raise ValueError("SED contains non-finite values.")
            except Exception as e:
                # Print a compact debug bundle to trace the bad region
                print("[SED ERROR]", str(e))
                print(f"  dist={dist:.3f}, teff1={teff1:.1f}, teff2={teff2:.1f}, "
                    f"r1={requiv1:.4f}, r2={requiv2:.4f}, incl={incl:.3f}")
                # Force the likelihood to reject this point gracefully
                return None, None, None, None
    else: 
        sed_model = None 
            
    return y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model 

def softplus(x):
    # numerically stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def lnlikelihood(params, data_dict, C, period, t0_ref, ecc_bool, rv_bool, use_ellc, model_phases,
                 lc_coeff=1, rv_coeff=1, sed_coeff=1):
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist, vgamma, ecc_val, per0_rad = transform_params(
        params, period, rv_bool, ecc_bool, t0_ref=t0_ref
    )

    try:
        y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model = forward_model(
            params, data_dict, C, period, t0_ref, ecc_bool, rv_bool, use_ellc, model_phases=model_phases
        )
        if y_pred_lc is None:
            return -np.inf
    except ValueError as e:
        print("Catching exception.")
        print(e)
        sys.stdout.flush()
        return -np.inf

    if sed_model is not None:
        eta_alpha_sed, eta_sigma_lc = params[8:10]
    else:
        eta_alpha_sed = None
        eta_sigma_lc = None
    
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
        if eta_sigma_lc is not None:
            jitter =softplus(eta_sigma_lc)    # additive (relative flux)
            var = sigma_lc_obs**2 + jitter**2
        else:
            var = sigma_lc_obs**2
        chi2_lc += np.sum((data_lc - y_pred) ** 2 / var) + np.sum(np.log(2*np.pi*var))
        N_lc_points += len(data_lc)

    # Calculate chi-squared for RVs, if present
    chi2_rv = 0
    N_rv_points = 0
    rv_params = 5 # m1, m2, i, a, vgamma
    if ecc_bool:
        rv_params += 3 # ecc, per0, t0_supconj
    if y_pred_rv_primary is not None and y_pred_rv_secondary is not None and rv_dataset is not None:
        sigma_rv1 = data_dict[rv_dataset]["primary_sigmas"]
        sigma_rv2 = data_dict[rv_dataset]["secondary_sigmas"]
        data_rv1 = data_dict[rv_dataset]["primary"]
        data_rv2 = data_dict[rv_dataset]["secondary"]
        chi2_rv += np.sum((data_rv1 - y_pred_rv_primary) ** 2 / sigma_rv1**2)
        chi2_rv += np.sum((data_rv2 - y_pred_rv_secondary) ** 2 / sigma_rv2**2)
        N_rv_points = len(data_rv1) + len(data_rv2)

    # Calculate chi-squared for SED, if present
    chi2_sed = 0
    N_sed_points = 0
    if "sed" in data_dict and sed_model is not None:
        data_sed = data_dict["sed"]["fluxes"]
        sigma_sed = data_dict["sed"]["flux_errs"]
        alpha_floor = 0.02
        alpha_cap   = 0.15
        alpha_sed   = alpha_floor + (alpha_cap - alpha_floor) * (1.0 / (1.0 + np.exp(-eta_alpha_sed)))
        # alpha_sed = alpha_floor + softplus(eta_alpha_sed)
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

    print(f"Scaled NLL terms: LC - {chi2_lc * w_LC}, RV - {chi2_rv * w_RV}, SED - {chi2_sed * w_SED}")
    # print(f"Reduced Chi2: LC - {chi2_lc}, RV - {chi2_rv}, SED - {chi2_sed}")

    # 1e4 works well
    # w_LC=0.4
    # w_RV=0.6
    # w_SED=0.6
    # chi2 = chi2_lc + chi2_rv + chi2_sed
    chi2 = w_LC * chi2_lc + w_RV * chi2_rv + w_SED * chi2_sed
    return -0.5 * chi2
