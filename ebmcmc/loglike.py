import phoebe
import numpy as np
from binarysed.binarysed import SED
import sys
import time
import logging
from scipy.interpolate import interp1d

MODEL_TEMPLATE = None

_WORKER_STATE = {}

def _pool_init(data_dict, model_phases, use_ellc):
    """Initializer for multiprocessing Pool: build a bundle once per process."""
    b, sed_obj = _build_template_bundle(data_dict, model_phases, use_ellc)
    _WORKER_STATE["b"] = b
    _WORKER_STATE["sed_obj"] = sed_obj

def _build_template_bundle(data_dict, model_phases, use_ellc):
    b = phoebe.default_binary()

    if model_phases is None: 
        model_phases = phoebe.linspace(0,0.9999,201)

    # Add datasets once; do NOT pass fluxes (they’re the *data*).
    for dataset in data_dict:
        if dataset.startswith("lc"):
            b.add_dataset(
                "lc",
                times=data_dict[dataset]["times"],
                sigmas=data_dict[dataset]["sigmas"],
                compute_phases=model_phases,
                dataset=dataset
            )
            if "tess".lower() in dataset.lower(): 
                b.set_value(f"passband@{dataset}", value = "TESS:T") 
            elif "sdss" in dataset.lower(): 
                b.set_value(f"passband@{dataset}", value="SDSS:g") 
            elif "johnson" in dataset.lower(): 
                b.set_value(f"passband@{dataset}", value="Johnson:V") 
            elif "ztf" in dataset.lower(): 
                b.set_value(f"passband@{dataset}", value="ZTF:r") 
            elif "kepler" in dataset.lower(): 
                b.set_value(f"passband@{dataset}", value="Kepler:mean")
        elif dataset.startswith("rv"):
            b.add_dataset(
                "rv",
                times={
                    "primary":  data_dict[dataset]["primary_times"],
                    "secondary":data_dict[dataset]["secondary_times"],
                }, 
                rvs={ 
                    "primary": data_dict[dataset]["primary"], 
                    "secondary": data_dict[dataset]["secondary"], 
                },
                sigmas={
                    "primary":  data_dict[dataset]["primary_sigmas"],
                    "secondary":data_dict[dataset]["secondary_sigmas"],
                },
                passband="Bolometric:900-40000"
            )

    b.set_value_all("ld_mode", "lookup" if use_ellc else "interp")
    b.set_value_all("ntriangles", value=1500)

    if use_ellc and 'ellcbackend' not in b.computes:
        b.add_compute('ellc', compute='ellcbackend')

    b.flip_constraint("mass@primary", solve_for="sma@binary@component")

    sed_obj = SED(data_dict["sed"])

    return b, sed_obj

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


def lnprob(params, data_dict, logit_q_init, asini_init, period_init, log_dist_init, t0, log_Msum_init,
            C, ecc_bool, rv_bool, eclipsing, use_ellc,
           lc_coeff, rv_coeff, sed_coeff, model_phases, A_obs, sigma_A):
    """
    Computes the log-probability by combining the log-prior and the log-likelihood.
    
    Args:
        params (list): A list of model parameters.
        data_dict (dict): A dictionary containing the observed data.
        q_init (float): Initial estimate for q.
        period_init (float): Initial estimate for period.
    
    Returns:
        float: The combined log-probability.
    """
    start_time = time.time()
    # print(f"lnprob called with {params}")
    sys.stdout.flush()
    # try:
    lp = lnprior(params, logit_q_init, asini_init, period_init, log_dist_init, log_Msum_init, C, ecc_bool, rv_bool, eclipsing, A_obs, sigma_A)
    if not np.isfinite(lp):
        return -np.inf
    elapsed_time = time.time() - start_time
    # print(f"lnprob completed in {elapsed_time:.5f} seconds: lp={lp}")
    sys.stdout.flush()
    return lp + lnlikelihood(params, data_dict, C, period_init, t0, ecc_bool, rv_bool, use_ellc, 
                                lc_coeff=lc_coeff, rv_coeff=rv_coeff, sed_coeff=sed_coeff,
                                model_phases=model_phases)
    # except Exception as e:
    #     print(f"lnprob failed: {e}")
    #     sys.stdout.flush()
    #     return -np.inf

def sigmoid(z): 
        return 1/(1+np.exp(-z))

def transform_params(params, period, rv_bool, ecc_bool):
    
    logit_q, log_Msum, log_teff1, log_tefffrac = params[:4]
    log_rfrac, logit_rsumfrac, logit_cosi, log_dist = params[4:8]
    eta_alpha_sed, eta_sigma_lc = params[8:10]

    q = sigmoid(logit_q)
    #log_Msum = 3 * log_requiv1 + C + ridge_scale
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

    G = 2942.206217504419328179210424423218 # Gravitational constant in solar units
    R_sun = 695700 # km

    a = (Msum * period**2 * G / (4 * np.pi**2))**(1/3)  # Semi-major axis in solar radii
    a1 = a / (1/q + 1)
    M1 = Msum/(1+q)
    M2 = M1*q
    asini = a * np.sin(np.radians(incl))

    requiv1 = r1frac * a
    requiv2 = r2frac * a

    # if rv_bool:
    #     vgamma = params[9]
    #     if not (-200 < vgamma < 200):
    #         return -np.inf
    # if ecc_bool:
    #     (ecc, per0) = params[10:12]
    #     pblums = params[12:]
    #     if not (0 < ecc < 1):
    #         return -np.inf
    #     if not (0 < per0 < 360):
    #         return -np.inf
    # else:
    #     pblums = params[10:]

    return q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist

def lnprior(params, logit_q_init, asini_init, period, log_dist_init, log_Msum_init, C, ecc_bool, rv_bool, eclipsing, A_obs, sigma_A):
    """
    Defines the log-prior function for the parameters.
    
    Args:
        params (list): A list of model parameters
    
    Returns:
        float: The log-prior value. Returns -∞ for parameters outside valid bounds.
    """
    # Unpack parameters
    # (teffratio, incl, requivsumfrac, requiv_secondary, q, t0_supconj, asini,
    #  teff_secondary, period) = params[:9]
    # (q, Msum, period, t0_supconj, teff1, teff2, requiv1, requiv2, incl) = params[:9]
    # log_q, log_Msum, period, t0_supconj, teff1, teff2, log_requiv1, log_requiv2, incl = params[:9]
    tp = transform_params(params, period, rv_bool, ecc_bool)
    if not isinstance(tp, tuple):
        return -np.inf
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist = tp
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
    RL1, RL2 = roche_lobe_frac(q)

    # keep the stars below ~85% of their RLs (softly)
    log_prior_roche1 = soft_barrier(requiv1/a, upper=0.85*RL1, k=10.0)
    log_prior_roche2 = soft_barrier(requiv2/a, upper=0.85*RL2, k=10.0)

    A_model = ( q*(requiv1/a)**3 + (requiv2/a)**3 ) * (np.sin(np.radians(incl))**2)
    log_prior_amplitude = soft_barrier(A_model, lower=0.5*A_obs, k=5.0)

    
    if not (0 < incl < 90):
        # print(f"incl value: {incl}")
        return -np.inf
    mu_logd = log_dist_init
    sigma_logd = 0.25
    log_prior_dist = -0.5 * ((np.log(dist) - mu_logd) / sigma_logd)**2
    log_prior_dist_max = soft_barrier(dist, upper=1200.0, k=0.8)
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

    # Uniform priors return 0 (log(1)); if Gaussian, use -0.5 * ((param - mu)/sigma)**2
    q_init = sigmoid(logit_q_init)
    sigma_logit_q = 0.35    # ~weak; tune 0.2–0.6 if needed
    log_prior_q = -0.5 * ((logit(q) - logit(q_init)) / sigma_logit_q)**2
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

    # optional soft safety net instead of a hard cap
    log_prior_msum_upper = soft_barrier(Msum, upper=15.0, k=3.0)

    A_model = ( q*(requiv1/a)**3 + (requiv2/a)**3 ) * (np.sin(np.radians(incl))**2)

    # log-space weak match to observed semi-amplitude
    log_prior_amplitude = -0.5 * ((np.log(A_model + 1e-12) - np.log(A_obs)) / 0.5)**2

    # Gaussian prior on asini   
    asini = a * np.sin(np.radians(incl))
    log_prior_asini = -0.5 * ((asini - asini_init) / (0.03*asini_init))**2
    log_prior_total = log_prior_q + log_prior_asini + log_prior_roche1 + log_prior_roche2 + log_prior_a + log_prior_msum + log_prior_msum_upper + log_prior_dist + log_prior_dist_max + log_prior_amplitude + log_prior_noneclipse#+ log_prior_ridge#+ log_prior_K1 + log_prior_K2

    if eclipsing:
        log_prior_total += log_prior_incl

    alpha_floor = 0.03
    alpha_cap   = 0.4

    eta_alpha_sed = params[8]
    tilde = 1/(1+np.exp(-eta_alpha_sed))                 # (0,1)
    alpha_sed = alpha_floor + (alpha_cap - alpha_floor) * tilde

    # mild Beta prior on tilde (discourages living at the cap/floor)
    a, b = 2.0, 5.0
    log_prior_alpha = (a-1)*np.log(tilde + 1e-12) + (b-1)*np.log(1-tilde + 1e-12)
    log_prior_total += log_prior_alpha

    return log_prior_total

def forward_model(params, data_dict, C, period, t0, ecc_bool, rv_bool, use_ellc, model_phases=None):
    # Decode params (unchanged)
    q, Msum, teff1, teff2, requiv1, requiv2, a, incl, dist = transform_params(params, period, rv_bool, ecc_bool)

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

    if ecc_bool:
        # set your ecc/per0 when you wire them
        b.set_value("ecc@binary@component", 0.0)
        b.set_value("per0@binary@component", 0.0)

    b.set_value("gravb_bol@primary",   value=(0.9  if teff1 > 8000 else 0.32))
    b.set_value("irrad_frac_refl_bol@primary",   value=(1.0  if teff1 > 8000 else 0.6))
    b.set_value("gravb_bol@secondary", value=(0.9  if teff2 > 8000 else 0.32))
    b.set_value("irrad_frac_refl_bol@secondary", value=(1.0  if teff2 > 8000 else 0.6))

    # limb-darkening source: set *both* branches explicitly
    logg_primary  = b.get_value("logg@primary@component")
    logg_secondary= b.get_value("logg@secondary@component")

    src1 = 'phoenix' if (teff1 < 3000 or logg_primary  > 5) else 'ck2004'
    src2 = 'phoenix' if (teff2 < 3000 or logg_secondary> 5) else 'ck2004'
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

def lnlikelihood(params, data_dict, C, period, t0, ecc_bool, rv_bool, use_ellc, model_phases,
                 lc_coeff=1, rv_coeff=1, sed_coeff=1):
    """
    Computes the log-likelihood for the given model parameters and observed data.

    Args:
        params (list): A list of model parameters (e.g., period, inclination, temperatures).
        data_dict (dict): A dictionary containing the observed data (light curves, RVs).
        ecc_bool (bool): Whether eccentricity is being fit.
        rv_bool (bool): Whether radial velocities are being fit.
        use_ellc (bool): Whether to use the ellc backend.
        lc_coeff (float): Weight coefficient for light curve contribution.
        rv_coeff (float): Weight coefficient for radial velocity contribution.
        sed_coeff (float): Weight coefficient for SED contribution.

    Returns:
        float: The computed log-likelihood value.
    """

    try:
        # print('Calculating likelihood...')
        sys.stdout.flush()
        y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model = forward_model(params, 
                                                                                        data_dict, 
                                                                                        C,
                                                                                        period,
                                                                                        t0,
                                                                                        ecc_bool, 
                                                                                        rv_bool, 
                                                                                        use_ellc,
                                                                                        model_phases=model_phases)   
        # print('Successful computation.')
        if y_pred_lc is None:
            return -np.inf
        sys.stdout.flush()
    except ValueError as e:
        print("Catching exception.")
        print(e)
        sys.stdout.flush()
        return -np.inf

    eta_alpha_sed, eta_sigma_lc = params[8:10]
    
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
        
        sigma_floor_lc = max(1e-3, np.nanmedian(sigma_lc_obs))  # e.g., 3e-4–8e-4                                       # ~3% fractional SED jitter

        # map to physical jitters with floors
        sigma_lc  = sigma_floor_lc + softplus(eta_sigma_lc)    # additive (relative flux)

        var = sigma_lc_obs**2 + sigma_lc**2
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
