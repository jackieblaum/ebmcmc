import phoebe
import numpy as np
import binarysed

phoebe.mpi_off()

def lnprob(params, data_dict, q_init, period_init, sigma_lnf_range, t0_range, ecc_bool):
    """
    Computes the log-probability by combining the log-prior and the log-likelihood.
    
    Args:
        params (list): A list of model parameters.
        data_dict (dict): A dictionary containing the observed data.
        q_init (float): Initial estimate for q.
        period_init (float): Initial estimate for period.
        sigma_lnf_range (tuple): Range for sigma_lnf.
    
    Returns:
        float: The combined log-probability.
    """
    lp = lnprior(params, q_init, period_init, sigma_lnf_range, t0_range, ecc_bool)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikelihood(params, data_dict)

def lnprior(params, q_init, period_init, sigma_lnf_range, t0_range, ecc_bool):
    """
    Defines the log-prior function for the parameters.
    
    Args:
        params (list): A list of model parameters
    
    Returns:
        float: The log-prior value. Returns -∞ for parameters outside valid bounds.
    """
    # Unpack parameters
    (teffratio, incl, requivsumfrac, requiv_secondary, q, t0_supconj, asini,
     teff_secondary, period, sigma_lnf) = params[:10]
    if ecc_bool:
        (ecc, per0) = params[10:12]
        pblums = params[12:]
        if not (0 < ecc < 1):
            return -np.inf
        if not (0 < per0 < 360):
            return -np.inf
    else:
        pblums = params[10:]

    # Check priors
    if not (0 < teffratio <= 1.2):
        return -np.inf
    if not (0 < incl < 90):
        return -np.inf
    if not (0 < q <= 1):
        return -np.inf
    if not (1e-6 < period):
        return -np.inf
    if not (sigma_lnf_range[0] < sigma_lnf < sigma_lnf_range[1]):
        return -np.inf
    if not (teff_secondary < 300):
        return -np.inf
    if not (t0_range[0] < t0_supconj < t0_range[1]):
        return -np.inf
    if not (np.all(pblums) > 0):
        return -np.inf

    # More priors as needed for other parameters
    # Uniform priors return 0 (log(1)); if Gaussian, use -0.5 * ((param - mu)/sigma)**2
    log_prior_q = -0.5 * ((q - q_init) / (q_init * 0.1))**2  # Gaussian prior with mean q_init and std dev 0.1 * q_init
    log_prior_period = -0.5 * ((period - period_init) / (0.1 * period_init))**2

    return log_prior_q + log_prior_period

def forward_model(params, data_dict):

    # Unpack the input parameters
    (
        teffratio,
        incl,
        requivsumfrac,
        requiv_secondary,
        q,
        t0_supconj,
        asini,
        teff_secondary,
        period,
        _,
    ) = params[:10]
    pblums = params[10:]

    if len(params) > 12:
        ecc, per0 = params[10:12]
    else:
        ecc, per0 = 0, 0

    # Create a new PHOEBE bundle and set the parameters
    b = phoebe.default_binary()

    # Add the datasets (LCs and RVs) from data_dict to the PHOEBE bundle
    for dataset in data_dict:
        if dataset.startswith("lc"):
            b.add_dataset(
                "lc",
                times=data_dict[dataset]["times"],
                fluxes=data_dict[dataset]["data"],
                sigmas=data_dict[dataset]["sigmas"],
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

    b.set_value_all('ld_mode', 'lookup')
    # Set the PHOEBE parameters
    b.flip_constraint("teffratio@binary@constraint", solve_for="teff@primary@component")
    b.flip_constraint(
        "requivsumfrac@binary@constraint", solve_for="requiv@primary@component"
    )
    b.flip_constraint("asini@binary@constraint", solve_for="sma@binary@component")

    b.set_value("teffratio@binary@component", teffratio)
    b.set_value("incl@binary@component", incl)
    b.set_value("requivsumfrac@binary@component", requivsumfrac)
    b.set_value("requiv@secondary@component", requiv_secondary)
    b.set_value("q@binary@component", q)
    b.set_value("t0_supconj@binary@component", t0_supconj)
    b.set_value("asini@binary@component", asini)
    b.set_value("teff@secondary@component", teff_secondary)
    b.set_value("period@binary@component", period)

    if teff_secondary > 8000:
        b.set_value("gravb_bol@secondary", value=0.9)
        b.set_value("irrad_frac_refl_bol@secondary", value=1.0)
    teff_primary = b.get_value("teff@primary@component")
    if teff_primary > 8000:
        b.set_value("gravb_bol@primary", value=0.9)
        b.set_value("irrad_frac_refl_bol@primary", value=1.0)

    logg_primary = b.get_value("logg@primary@component")
    logg_secondary = b.get_value("logg@secondary@component")

    if teff_primary < 3000 or logg_primary > 5:
        b.set_value('ld_coeffs_source@primary', value='phoenix')
    if teff_secondary < 3000 or logg_secondary > 5:
        b.set_value('ld_coeffs_source@secondary', value='phoenix')

    # Set eccentricity and periastron, if needed
    if len(params) > 12:
        b.set_value("ecc@binary@component", ecc)
        b.set_value("per0@binary@component", per0)

    # Set pblum values for each light curve
    for dataset, pblum in zip(b.datasets, pblums):
        if dataset.startswith("lc"):
            b.set_value(f"pblum@primary@{dataset}@dataset", pblum)

    # Run PHOEBE computation
    b.add_compute("ellc", compute="fastcompute")
    b.run_compute(compute="fastcompute")

    # Get model predictions for light curves (LCs) and radial velocities (RVs)
    y_pred_lc = [
        b.get_value(f"fluxes@model@{dataset}")
        for dataset in b.datasets
        if dataset.startswith("lc")
    ]
    y_pred_rv_primary = (
        b.get_value(f"rvs@model@{dataset}@primary") if "rv" in b.datasets else None
    )
    y_pred_rv_secondary = (
        b.get_value(f"rvs@model@{dataset}@secondary") if "rv" in b.datasets else None
    )

    if "sed" in data_dict:
        sed_obj = binarysed.SED(data_dict["sed"])
        wavelengths = data_dict["sed"]["wavelengths"]
        teff_primary = b.get_value("teff@primary@component")
        teff_secondary = b.get_value("teff@secondary@component")
        requiv_primary = b.get_value("requiv@primary@component")
        requiv_secondary = b.get_value("requiv@secondary@component")
        logg1 = b.get_value("logg@primary@component")
        logg2 = b.get_value("logg@secondary@component")

        sed_model = sed_obj.create_apparent_sed(
            wavelengths,
            teff_primary,
            teff_secondary,
            requiv_primary,
            requiv_secondary,
            logg1,
            logg2,
            select_wavelengths=True
        )
    else:
        sed_model = None

    return y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model


def lnlikelihood(params, data_dict):
    """
    Computes the log-likelihood for the given model parameters and observed data.

    Args:
        params (list): A list of model parameters (e.g., period, inclination, temperatures).
        data_dict (dict): A dictionary containing the observed data (light curves, RVs, SED).

    Returns:
        float: The computed log-likelihood value.
    """

    try:
        y_pred_lc, y_pred_rv_primary, y_pred_rv_secondary, sed_model = forward_model(params, data_dict)   
        print('Successful computation.')
    except ValueError as e:
        print("Catching exception.")
        print(e)
        return -np.inf
    
    sigma_lnf = params[9]

    # Calculate chi-squared for light curves
    chi2_lc = 0
    for dataset, y_pred in zip(data_dict, y_pred_lc):
        data_lc = data_dict[dataset]["data"]
        sigma_lc = data_dict[dataset]["sigmas"]
        sigma_lc_sq = sigma_lc**2 + y_pred**2 * np.exp(2 * sigma_lnf)
        chi2_lc += np.sum(np.log(sigma_lc_sq) + (data_lc - y_pred) ** 2 / sigma_lc_sq) / len(data_lc)

    # Calculate chi-squared for RVs, if present
    chi2_rv = 0
    if y_pred_rv_primary is not None and y_pred_rv_secondary is not None:
        sigma_rv1 = data_dict[dataset]["primary_sigmas"]
        sigma_rv2 = data_dict[dataset]["secondary_sigmas"]
        data_rv1 = data_dict[dataset]["primary"]
        data_rv2 = data_dict[dataset]["secondary"]
        chi2_rv += np.sum(
            np.log(sigma_rv1**2) + (data_rv1 - y_pred_rv_primary) ** 2 / sigma_rv1**2
        ) / len(data_rv1)
        chi2_rv += np.sum(
            np.log(sigma_rv2**2) + (data_rv2 - y_pred_rv_secondary) ** 2 / sigma_rv2**2
        ) / len(data_rv2)

    # Calculate chi-squared for SED, if provided
    chi2_sed = 0
    if sed_model is not None:
        obs_fluxes = data_dict["sed"]["fluxes"]
        obs_flux_errs = data_dict["sed"]["flux_errs"]

        chi2_sed = np.sum(
            np.log(obs_flux_errs**2) + (obs_fluxes - sed_model) ** 2 / obs_flux_errs**2
        ) / len(obs_fluxes)

    # Return the total log-likelihood
    # print(f"Chi2: LC - {chi2_lc}, RV - {chi2_rv}, SED - {chi2_sed}")
    chi2 = chi2_lc + chi2_rv + chi2_sed
    return -0.5 * chi2
