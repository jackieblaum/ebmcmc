import phoebe
import numpy as np
import binarysed
import sys
import time
import logging
from scipy.interpolate import interp1d

def lnprob(params, data_dict, q_init, asini_init, period_init, Msum_init, t0_init, ecc_bool, rv_bool, eclipsing, use_ellc,
           lc_coeff, rv_coeff, sed_coeff):
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
    print(f"lnprob called with {params}")
    sys.stdout.flush()
    try:
        lp = lnprior(params, q_init, asini_init, period_init, Msum_init, t0_init, ecc_bool, rv_bool, eclipsing)
        if not np.isfinite(lp):
            raise ValueError(f'log prior not finite: {lp}')
        elapsed_time = time.time() - start_time
        print(f"lnprob completed in {elapsed_time:.5f} seconds: lp={lp}")
        sys.stdout.flush()
        return lp + lnlikelihood(params, data_dict, ecc_bool, rv_bool, use_ellc, 
                                 lc_coeff=lc_coeff, rv_coeff=rv_coeff, sed_coeff=sed_coeff)
    except Exception as e:
        print(f"lnprob failed: {e}")
        sys.stdout.flush()
        return -np.inf

def lnprior(params, q_init, asini_init, period_init, Msum_init, t0_init, ecc_bool, rv_bool, eclipsing):
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
    (q, Msum, period, t0_supconj, teff1, teff2, requiv1, requiv2, incl) = params[:9]
    if rv_bool:
        vgamma = params[9]
        if not (-200 < vgamma < 200):
            print(f"vgamma value: {vgamma}")
            return -np.inf
    if ecc_bool:
        (ecc, per0) = params[10:12]
        pblums = params[12:]
        if not (0 < ecc < 1):
            return -np.inf
        if not (0 < per0 < 360):
            return -np.inf
    else:
        pblums = params[9:]

    G = 6.67430e-8            # Gravitational constant in cgs units
    a = (G * Msum * (period * 86400)**2 / (4 * np.pi**2))**(1/3)  # Semi-major axis in cm
    requivsumfrac = (requiv1 + requiv2)/a
    asini = a * np.sin(np.radians(incl))

    # Check priors
    if not (0 < q <= 1):
        print(f"q value: {q}")
        return -np.inf
    if not (0.1 < Msum < 500):  # Stellar mass range
        print(f"Msum value: {Msum}")
        return -np.inf
    if not (1e-6 < period < 1e6):
        print(f"period value: {period}")
        return -np.inf
    if not (2500 < teff1 < 50000):
        print(f"teff_secondary value: {teff1}")
        return -np.inf
    if not (2500 < teff2 < 50000):
        print(f"teff_secondary value: {teff2}")
        return -np.inf
    if not (1e-6 < requiv1 < a):
        print(f"requiv_secondary value: {requiv1}")
        return -np.inf
    if not (1e-6 < requiv2 < a):
        print(f"requiv_secondary value: {requiv2}")
        return -np.inf
    if not (0 < incl < 90):
        print(f"incl value: {incl}")
        return -np.inf
    # else:
    #     i_max_rad = np.arccos(requivsumfrac)
    #     i_max = np.degrees(i_max_rad)
    #     if not (0 < incl < i_max):
    #         print(f"requivsumfrac: {requivsumfrac}")
    #         print(f"i_max: {i_max}")
    #         print(f"incl value: {incl}")
    #         return -np.inf
    if not (0 < np.all(pblums) < 1e6):
        print(f"pblums value: {pblums}")
        return -np.inf

    # More priors for other parameters
    if eclipsing:
        log_prior_incl = np.log(np.sin(np.radians(incl)))  # sin(i) prior for random orientation

    # Uniform priors return 0 (log(1)); if Gaussian, use -0.5 * ((param - mu)/sigma)**2
    log_prior_q = -0.5 * ((q - q_init) / (q_init * 0.05))**2  # Gaussian prior with mean q_init and std dev 0.01 * q_init
    log_prior_period = -0.5 * ((period - period_init) / (0.0001 * period_init))**2
    log_prior_t0_supconj = -0.5 * ((t0_supconj - t0_init) / (0.01 * period_init))**2
    log_prior_msum = -0.5 * ((Msum - Msum_init) / (0.05 * Msum_init))**2

    # Gaussian prior on asini
    log_prior_asini = -0.5 * ((asini - asini_init) / (0.1*asini_init))**2
    log_prior_total = log_prior_q + log_prior_period + log_prior_t0_supconj + log_prior_asini

    if eclipsing:
        log_prior_total += log_prior_incl

    return log_prior_total

def forward_model(params, data_dict, ecc_bool, rv_bool, use_ellc):

    print('Running forward model...')
    sys.stdout.flush()
    # Unpack the input parameters
    (q, Msum, period, t0_supconj, teff1, teff2, 
     requiv1, requiv2, incl
    ) = params[:9]

    ecc_start = 9
    if rv_bool:
        vgamma = params[9]
        ecc_start = 10
    if ecc_bool:
        ecc, per0 = params[ecc_start:ecc_start+2]
        pblums = params[ecc_start+2:]
    else:
        ecc, per0 = 0, 0
        pblums = params[ecc_start:]

    # Create a new PHOEBE bundle and set the parameters
    b = phoebe.default_binary()

    model_phases = phoebe.linspace(0,1,201)
    # phoebe.mpi_on()
    print(f'nprocs: {phoebe.multiprocessing_get_nprocs()}')

    # b.add_compute('ellc', compute='fastcompute')

    # Add the datasets (LCs and RVs) from data_dict to the PHOEBE bundle
    for dataset in data_dict:
        if dataset.startswith("lc"):
            b.add_dataset(
                "lc",
                times=data_dict[dataset]["times"],
                fluxes=data_dict[dataset]["data"],
                sigmas=data_dict[dataset]["sigmas"],
                compute_phases=model_phases,
                dataset=dataset
            )
            # Set the limb darkening mode to 'lookup'
            # print(dataset)
            if "tess".lower() in dataset.lower():
                b.set_value(f"passband@{dataset}", value = "TESS:T")
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
            )
    if use_ellc:
        b.set_value_all('ld_mode', 'lookup')
    else:
        b.set_value_all('ld_mode', 'interp')
    b.set_value_all('ntriangles', value=1500)
    # Set the PHOEBE parameters
    b.flip_constraint("mass@primary", solve_for="sma@binary@component")
    
# q, Msum, period, t0_supconj, teff1, teff2, 
#      requiv1, requiv2, incl
    M1 = Msum / (1 + q)       # Primary mass
    b.set_value("q@binary@component", q)
    b.set_value("mass@primary@component", M1)
    b.set_value("period@binary@component", period)
    b.set_value("t0_supconj@binary@component", t0_supconj)
    b.set_value("teff@primary@component", teff1)
    b.set_value("teff@secondary@component", teff2)
    b.set_value("requiv@primary@component", requiv1)
    b.set_value("requiv@secondary@component", requiv2)
    b.set_value("incl@binary@component", incl)

    if teff1 > 8000:
        b.set_value("gravb_bol@primary", value=0.9)
        b.set_value("irrad_frac_refl_bol@primary", value=1.0)
    if teff2 > 8000:
        b.set_value("gravb_bol@secondary", value=0.9)
        b.set_value("irrad_frac_refl_bol@secondary", value=1.0)

    logg_primary = b.get_value("logg@primary@component")
    logg_secondary = b.get_value("logg@secondary@component")

    if teff1 < 3000 or logg_primary > 5:
        b.set_value_all('ld_coeffs_source@primary', value='phoenix')
    if teff2 < 3000 or logg_secondary > 5:
        b.set_value_all('ld_coeffs_source@secondary', value='phoenix')

    if rv_bool:
        b.set_value("vgamma@system", vgamma)

    # Set eccentricity and periastron, if needed
    if ecc_bool:
        b.set_value("ecc@binary@component", ecc)
        b.set_value("per0@binary@component", per0)

    # Set pblum values for each light curve
    for dataset, pblum in zip(b.datasets, pblums):
        if dataset.startswith("lc"):
            b.set_value(f"pblum@primary@{dataset}@dataset", pblum)

    # Run PHOEBE computation
    # b.add_compute("ellc", compute="fastcompute")
    sys.stdout.flush()
    if use_ellc:
        if not 'ellcbackend' in b.computes:
            b.add_compute('ellc', compute='ellcbackend')
        b.run_compute(compute='ellcbackend')
    else:
        print("Running PHOEBE computation...")
        b.run_compute(compute='phoebe01')
        print("Finished PHOEBE computation.")
        sys.stdout.flush()
    # Get model predictions for light curves (LCs) and radial velocities (RVs)
    y_pred_lc = []
    for dataset in b.datasets:
        if dataset.startswith("lc"):
            # Get model fluxes and times
            model_fluxes = b.get_value(f"fluxes@model@{dataset}")
            model_times = b.get_value(f"times@model@{dataset}")

            # Convert to phase
            model_phases = b.to_phase(model_times)
            observed_times = b.get_value(f"times@{dataset}@dataset")
            observed_phases = b.to_phase(observed_times)

            # Ensure model_phases and model_fluxes are sorted
            sort_idx = np.argsort(model_phases)  # Get sorting order
            model_phases_sorted = np.array(model_phases)[sort_idx]
            model_fluxes_sorted = np.array(model_fluxes)[sort_idx]

            # Create interpolation function
            interp_func = interp1d(model_phases_sorted, model_fluxes_sorted, kind="cubic", fill_value="extrapolate")

            # Interpolate at the actual observed times
            interpolated_fluxes = interp_func(observed_phases)

            # Unsort back to original order of observed phases/times
            # inverse_idx = np.argsort(sort_idx)
            # unsorted_interpolated_fluxes = interpolated_fluxes[inverse_idx]
            y_pred_lc.append(interpolated_fluxes)
    
    y_pred_rv_primary = (
        b.get_value(f"rvs@model@{dataset}@primary") if "rv" in b.datasets else None
    )
    y_pred_rv_secondary = (
        b.get_value(f"rvs@model@{dataset}@secondary") if "rv" in b.datasets else None
    )

    if "sed" in data_dict:
        if np.isnan(data_dict["sed"]["dist"]):
            print("No distance available; leaving SED out of the fit.")
            sed_model = None
        else:
            sed_obj = binarysed.SED(data_dict["sed"])
            wavelengths = data_dict["sed"]["wavelengths"]
            logg1 = b.get_value("logg@primary@component")
            logg2 = b.get_value("logg@secondary@component")

            sed_model = sed_obj.create_apparent_sed(
                wavelengths,
                teff1,
                teff2,
                requiv1,
                requiv2,
                logg1,
                logg2,
                select_wavelengths=True
        )
    else:
        sed_model = None

    return y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model


def lnlikelihood(params, data_dict, ecc_bool, rv_bool, use_ellc, 
                 lc_coeff=1, rv_coeff=1, sed_coeff=1):
    """
    Computes the log-likelihood for the given model parameters and observed data.

    Args:
        params (list): A list of model parameters (e.g., period, inclination, temperatures).
        data_dict (dict): A dictionary containing the observed data (light curves, RVs, SED).

    Returns:
        float: The computed log-likelihood value.
    """

    try:
        print('Calculating likelihood...')
        sys.stdout.flush()
        y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model = forward_model(params, data_dict, ecc_bool, rv_bool, use_ellc)   
        print('Successful computation.')
        sys.stdout.flush()
    except ValueError as e:
        print("Catching exception.")
        print(e)
        sys.stdout.flush()
        return -np.inf
    
    # Calculate chi-squared for light curves
    chi2_lc = 0
    N_lc_points = 0
    lc_params = 7 # q, r1, r2, P, t0_supconj, i, teffratio
    if ecc_bool:
        lc_params += 2 # ecc, per0
    for dataset, y_pred in zip(data_dict, y_pred_lc):
        data_lc = data_dict[dataset]["data"]
        sigma_lc = data_dict[dataset]["sigmas"]
        sigma_lc_sq = sigma_lc**2 # + y_pred**2 * np.exp(2 * sigma_lnf)
        chi2_lc += np.sum((data_lc - y_pred) ** 2 / sigma_lc_sq)
        N_lc_points += len(data_lc)
        lc_params += 1 # for pblum

    # Calculate chi-squared for RVs, if present
    chi2_rv = 0
    N_rv_points = 0
    rv_params = 5 # m1, m2, i, a, vgamma
    if ecc_bool:
        rv_params += 3 # ecc, per0, t0_supconj
    if y_pred_rv_primary is not None and y_pred_rv_secondary is not None:
        sigma_rv1 = data_dict[dataset]["primary_sigmas"]
        sigma_rv2 = data_dict[dataset]["secondary_sigmas"]
        data_rv1 = data_dict[dataset]["primary"]
        data_rv2 = data_dict[dataset]["secondary"]
        chi2_rv += np.sum((data_rv1 - y_pred_rv_primary) ** 2 / sigma_rv1**2)
        chi2_rv += np.sum((data_rv2 - y_pred_rv_secondary) ** 2 / sigma_rv2**2)
        N_rv_points = len(data_rv1) + len(data_rv2)

    # Calculate chi-squared for SED, if provided
    chi2_sed = 0
    sed_params = 6 # teff1, teff2, r1, r2, logg1, logg2
    N_sed_points = 0
    if sed_model is not None:
        obs_fluxes = data_dict["sed"]["fluxes"]
        obs_flux_errs = data_dict["sed"]["flux_errs"]

        chi2_sed = np.sum((obs_fluxes - sed_model) ** 2 / obs_flux_errs**2)
        N_sed_points = len(obs_fluxes)

    # Return the total log-likelihood
    w_LC = 0
    w_SED = 0
    w_RV = 0
    if N_lc_points is not 0:
        w_LC = (N_sed_points + N_rv_points) / N_lc_points
    if N_sed_points is not 0:
        w_SED = (N_lc_points + N_rv_points) / N_sed_points
    if N_rv_points is not 0:
        w_RV = (N_lc_points + N_sed_points) / N_rv_points
    print(f"Reduced Chi2: LC - {chi2_lc * w_LC}, RV - {chi2_rv * w_RV}, SED - {chi2_sed * w_SED}")
    # reduced_chi2_lc = chi2_lc/(N_lc_points - lc_params)
    # reduced_chi2_rv = chi2_rv/(N_rv_points - rv_params)
    # reduced_chi2_sed = chi2_sed/(N_sed_points - sed_params)
    # chi2 = lc_coeff * reduced_chi2_lc + rv_coeff * reduced_chi2_rv + sed_coeff * reduced_chi2_sed
    chi2 = w_LC * chi2_lc + w_RV * chi2_rv + w_SED * chi2_sed
    return -0.5 * chi2
