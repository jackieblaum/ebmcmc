import numpy as np
import aesara
import phoebe


class Loglike(aesara.tensor.Op):
    """
    Theano Op class for integrating a custom log-likelihood function with PyMC3.

    Attributes:
        data_dict (dict): A dictionary containing the observed data.
        loglike_grad (LoglikeGrad): An instance of the LoglikeGrad class for gradient calculation.
    """

    itypes = [aesara.tensor.dvector]  # Input types (list of parameters as a vector)
    otypes = [aesara.tensor.dscalar]  # Output types (log-likelihood as a scalar)

    def __init__(self, data_dict):
        """
        Initializes the Loglike Op with a data dictionary.

        Args:
            data_dict (dict): A dictionary containing the observed data.
        """
        self.data_dict = data_dict
        self.loglike_grad = LoglikeGrad(data_dict)

    def perform(self, node, inputs, outputs):
        """
        Performs the log-likelihood calculation.

        Args:
            node: Theano node containing information about the function call.
            inputs: List of input parameters for the log-likelihood function.
            outputs: List of output objects to store the computed log-likelihood.
        """
        # Unpack the input parameters
        params = (
            inputs[0]
            if not isinstance(inputs[0], aesara.tensor.TensorVariable)
            else inputs[0].eval()
        )

        # Compute the log-likelihood using the custom function
        logp = lnlikelihood(params, self.data_dict)

        # Store the result in outputs
        outputs[0][0] = np.array(logp)

    def grad(self, inputs, grad_outputs):
        """
        Computes the gradient of the log-likelihood function with respect to the parameters.

        Args:
            inputs: List of input parameters for which gradients are to be computed.
            grad_outputs: List of gradients passed from PyMC3.

        Returns:
            list: Gradients of the log-likelihood with respect to the inputs.
        """
        (theta,) = inputs
        grads = self.loglike_grad(theta)
        return [grad_outputs[0] * grads]


class LoglikeGrad(aesara.tensor.Op):
    """
    Theano Op class for computing the gradient of the log-likelihood function.

    Attributes:
        data_dict (dict): A dictionary containing the observed data.
    """

    itypes = [aesara.tensor.dvector]  # Input types (list of parameters as a vector)
    otypes = [aesara.tensor.dvector]  # Output types (gradients as a vector)

    def __init__(self, data_dict):
        """
        Initializes the LoglikeGrad Op with a data dictionary.

        Args:
            data_dict (dict): A dictionary containing the observed data.
        """
        self.data_dict = data_dict

    def perform(self, node, inputs, outputs):
        """
        Computes the gradient of the log-likelihood function with respect to the input parameters.

        Args:
            node: Theano node containing information about the function call.
            inputs: List of input parameters for which the gradients are computed.
            outputs: List of output objects to store the computed gradients.
        """
        (theta,) = inputs

        # Compute the gradients using the custom derivative function
        grads = der_log_likelihood([theta], self.data_dict)

        # Store the result in outputs
        outputs[0][0] = grads


def lnlikelihood(params, data_dict):
    """
    Computes the log-likelihood for the given model parameters and observed data.

    Args:
        params (list): A list of model parameters (e.g., period, inclination, temperatures).
        data_dict (dict): A dictionary containing the observed data (light curves, RVs, SED).

    Returns:
        float: The computed log-likelihood value.
    """
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
        sigma_lnf,
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
            )
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
    try:
        b.run_compute(compute="fastcompute")
    except ValueError:
        return -np.inf

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

    # Calculate chi-squared for light curves
    chi2_lc = 0
    for dataset, y_pred in zip(data_dict, y_pred_lc):
        data_lc = data_dict[dataset]["data"]
        sigma_lc = data_dict[dataset]["sigmas"]
        sigma_lc_sq = sigma_lc**2 + y_pred**2 * np.exp(2 * sigma_lnf)
        chi2_lc += np.sum(np.log(sigma_lc_sq) + (data_lc - y_pred) ** 2 / sigma_lc_sq)

    # Calculate chi-squared for RVs, if present
    chi2_rv = 0
    if y_pred_rv_primary is not None and y_pred_rv_secondary is not None:
        sigma_rv1 = data_dict[dataset]["primary_sigmas"]
        sigma_rv2 = data_dict[dataset]["secondary_sigmas"]
        data_rv1 = data_dict[dataset]["primary"]
        data_rv2 = data_dict[dataset]["secondary"]
        chi2_rv += np.sum(
            np.log(sigma_rv1**2) + (data_rv1 - y_pred_rv_primary) ** 2 / sigma_rv1**2
        )
        chi2_rv += np.sum(
            np.log(sigma_rv2**2) + (data_rv2 - y_pred_rv_secondary) ** 2 / sigma_rv2**2
        )

    # Calculate chi-squared for SED, if provided
    chi2_sed = 0
    if "sed" in data_dict:
        sed_obj = sed.SED(data_dict["sed"])
        wavelengths = data_dict["sed"]["wavelengths"]
        obs_fluxes = data_dict["sed"]["fluxes"]
        obs_flux_errs = data_dict["sed"]["flux_errs"]
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
        )

        chi2_sed = np.sum(
            np.log(obs_flux_errs**2) + (obs_fluxes - sed_model) ** 2 / obs_flux_errs**2
        )

    # Return the total log-likelihood
    chi2 = chi2_lc + chi2_rv + chi2_sed
    return -0.5 * chi2


def der_log_likelihood(theta, data_dict):
    """
    Computes the gradient of the log-likelihood function using finite differences.

    Args:
        theta (list): A list of model parameters for which the gradient is needed.
        data_dict (dict): A dictionary containing the observed data.

    Returns:
        numpy.ndarray: The gradient of the log-likelihood with respect to the parameters.
    """

    def lnlike(params):
        return lnlikelihood(params, data_dict)

    # Finite difference step size
    eps = np.sqrt(np.finfo(float).eps)

    # Compute the gradient using finite differences
    grads = scipy.optimize.approx_fprime(
        theta[0], lnlike, [eps * max(1, np.abs(param)) for param in theta[0]]
    )
    return grads
