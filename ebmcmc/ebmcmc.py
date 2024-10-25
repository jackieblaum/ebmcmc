import phoebe
import pymc as pm
import numpy as np
import os
import pytensor.tensor as pt
import scipy.optimize
import arviz as az
import pyphot
from datetime import datetime
import matplotlib.pyplot as plt
import xarray as xr
import pickle
import logging
from tqdm import tqdm
import binarysed
from ebmcmc.loglike import Loglike


class EBMCMC:
    """
    A class for performing Markov Chain Monte Carlo (MCMC) sampling on binary star systems using PHOEBE and pymc.
    """

    def __init__(
        self, bundle, trace_dir=None, sed=None, datasets=None, eclipsing=True, ecc=True
    ):
        self.bundle = bundle
        self.sed = sed
        self.model = None
        self.min_time = 1e9
        self.max_time = 0
        self.data_dict = self.create_data_dict(datasets=datasets)
        self.eclipsing = eclipsing
        self.ecc = ecc
        self.trace_dir = trace_dir
        self.likelihood_computations = 0

        self.initialize_bundle()
        self.initialize_logging()

    def initialize_bundle(self):
        """Initializes PHOEBE bundle values."""
        self.bundle.set_value_all("ld_mode", "lookup")
        self.bundle.set_value_all("pblum_mode", "component-coupled")

    def initialize_logging(self):
        """Initializes logging for PHOEBE and pymc."""
        phoebe_logger = phoebe.logger(
            clevel=None, flevel="CRITICAL", filename="phoebe.log"
        )
        phoebe_logger.propagate = False
        phoebe.progressbars_off()

        pymc_logger = logging.getLogger("pymc")
        pymc_logger.setLevel(logging.INFO)
        pymc_logger.propagate = True

    def create_data_dict(self, datasets=None):
        data_dict = {}
        if datasets is None:
            datasets = self.bundle.datasets

        for dataset in datasets:
            if dataset.startswith("lc"):
                data_dict[dataset] = self.extract_light_curve_data(dataset)
            elif dataset.startswith("rv"):
                data_dict[dataset] = self.extract_rv_data(dataset)
            else:
                raise ValueError(f"Unrecognized dataset type: {dataset}")

        if self.sed:
            data_dict["sed"] = self.sed

        return data_dict

    def extract_light_curve_data(self, dataset):
        times = self.bundle.get_value(f"times@{dataset}@dataset")
        self.min_time, self.max_time = min(self.min_time, np.min(times)), max(
            self.max_time, np.max(times)
        )
        return {
            "data": self.bundle.get_value(f"fluxes@{dataset}@dataset"),
            "sigmas": self.bundle.get_value(f"sigmas@{dataset}"),
            "times": times,
        }

    def extract_rv_data(self, dataset):
        primary_times = self.bundle.get_value(f"times@{dataset}@primary@dataset")
        secondary_times = self.bundle.get_value(f"times@{dataset}@secondary@dataset")
        return {
            "primary": self.bundle.get_value(f"rvs@primary@{dataset}@dataset"),
            "secondary": self.bundle.get_value(f"rvs@secondary@{dataset}@dataset"),
            "primary_sigmas": self.bundle.get_value(f"sigmas@{dataset}@primary"),
            "secondary_sigmas": self.bundle.get_value(f"sigmas@{dataset}@secondary"),
            "primary_times": primary_times,
            "secondary_times": secondary_times,
        }

    def define_model(self):
        period_init = self.bundle.get_value("period@binary@component")
        m1 = self.bundle.get_value("mass@primary@component")
        m2 = self.bundle.get_value("mass@secondary@component")
        q_init = self.bundle.get_value("q@binary@component")
        incl_init = self.bundle.get_value("incl@binary@component")
        asini_init = self.bundle.get_value("asini@binary@component")
        requivsumfrac_init = self.bundle.get_value("requivsumfrac@binary@component")
        teffratio_init = self.bundle.get_value("teffratio@binary@component")
        teff_secondary_init = self.bundle.get_value("teff@secondary@component")
        ecc_init = self.bundle.get_value("ecc@binary@component")
        per0_init = self.bundle.get_value("per0@binary@component")
        pblums_init = [
            self.bundle.get_value(f"pblum@primary@{dataset}@dataset")
            for dataset in self.bundle.datasets
            if dataset.startswith("lc")
        ]

        if q_init > 1:
            q_init = m1 / m2
            requiv_secondary_init = self.bundle.get_value("requiv@primary@component")
        else:
            requiv_secondary_init = self.bundle.get_value("requiv@secondary@component")

        with pm.Model() as self.model:
            # Define priors
            period = pm.TruncatedNormal(
                "period@binary@component",
                lower=0,
                mu=period_init,
                sigma=period_init * 0.1,
            )
            q = pm.TruncatedNormal(
                "q@binary@component", lower=0.1, upper=1, mu=q_init, sigma=q_init * 0.1
            )
            incl = pm.Uniform(
                "incl@binary@component", lower=1, upper=90, initval=incl_init
            )
            asini = pm.TruncatedNormal(
                "asini@binary@component",
                lower=0.01,
                upper=1000,
                mu=asini_init,
                sigma=asini_init * 0.1,
            )
            sma = asini / np.sin(incl * (2 * np.pi) / 360)

            mass_primary = (
                39.478418
                * (asini / np.sin(incl * (2 * np.pi) / 360)) ** 3
                / (period**2 * (q + 1))
            )
            mass_secondary = (
                39.478418
                * (asini / np.sin(incl * (2 * np.pi) / 360)) ** 3
                / (period**2 * (1 / q + 1))
            )

            requiv_secondary = pm.Uniform(
                "requiv@secondary@component",
                lower=0.1,
                upper=3,
                initval=requiv_secondary_init,
            )
            requivsumfrac = pm.Uniform(
                "requivsumfrac@binary@component",
                lower=0.01,
                upper=0.5,
                initval=requivsumfrac_init,
            )

            teff_secondary = pm.Uniform(
                "teff@secondary@component",
                lower=3500,
                upper=50000,
                initval=teff_secondary_init,
            )
            teffratio = pm.Uniform(
                "teffratio@binary@component",
                lower=0.7,
                upper=1.2,
                initval=teffratio_init,
            )

            if self.ecc:
                ecc = pm.Beta("ecc@binary@component", alpha=1, beta=5, initval=ecc_init)
                per0_rad = pm.VonMises(
                    "per0@rad", mu=per0_init * (2 * np.pi) / 360, kappa=1
                )
                per0 = pm.Deterministic(
                    "per0@binary@component", 360 / (2 * np.pi) * per0_rad
                )
            else:
                ecc = 0
                per0 = 0

            sigma_lnf = pm.Uniform("sigma_lnf", lower=-15, upper=-1)

            fit_params = [
                teffratio,
                incl,
                requivsumfrac,
                requiv_secondary,
                q,
                period,
                sigma_lnf,
                teff_secondary,
            ]

            if self.ecc:
                fit_params.extend([ecc, per0])

            for i, pblum in enumerate(pblums_init):
                fit_params.append(
                    pm.TruncatedNormal(
                        f"pblum@primary@{i}@dataset", lower=0, mu=pblum, sigma=0.01
                    )
                )

            loglike = Loglike(self.data_dict)
            params = pt.as_tensor_variable(fit_params)
            for param in fit_params:
                print(f"Type of param: {type(param)}")

            # params = pt.stack(fit_params)
            pm.Potential("like", loglike(params))

    def sample(self, ndraws=1000, cores=4, tune_steps=1000, target_accept=0.9):
        if self.model is None:
            self.define_model()

        with self.model:
            trace = pm.sample(
                draws=ndraws, cores=cores, tune=tune_steps, target_accept=target_accept
            )
            self.save_trace(trace)

        return trace

    def save_trace(self, trace):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.trace_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        for chain_idx in range(trace.posterior.chain.size):
            filename = f"trace_chain_{chain_idx}.nc"
            az.to_netcdf(trace.sel(chain=chain_idx), os.path.join(run_dir, filename))

    # Other functions remain the same, adjusted for updated syntax where needed


    def load_full_trace_states(self, truncate=False):
        """
        Loads the full trace state from previously saved runs.

        Args:
            truncate (bool, optional): If True, truncates the chains to the shortest length if they are of different lengths. Defaults to False.

        Returns:
            list or az.InferenceData: A list of individual chains or a combined InferenceData object, depending on the number of chains.
        """
        if not self.trace_dir:
            raise ValueError("Trace directory not provided")

        subdirs = [
            d
            for d in os.listdir(self.trace_dir)
            if os.path.isdir(os.path.join(self.trace_dir, d))
        ]
        sorted_dirs = sorted(
            subdirs, key=lambda x: os.path.getctime(os.path.join(self.trace_dir, x))
        )

        all_chains = []
        for subdir in sorted_dirs:
            chain_list = []
            indices = []
            for trace_file in os.listdir(os.path.join(self.trace_dir, subdir)):
                if trace_file.endswith(".nc"):
                    chain_idx = int(trace_file.split("_")[-1].split(".")[0])
                    indices.append(chain_idx)
                    single_chain = az.from_netcdf(
                        os.path.join(self.trace_dir, subdir, trace_file)
                    )
                    chain_list.append(single_chain.posterior)

            # Sort chains by indices to ensure correct order
            sorted_indices = np.argsort(indices)
            chain_list_sorted = [chain_list[i] for i in sorted_indices]
            all_chains.append(chain_list_sorted)

        if not all_chains:
            return None

        # Check if all chains have the same number of draws
        total_draws = np.array(
            [chain.sizes["draw"] for chain_set in all_chains for chain in chain_set]
        )
        if len(set(total_draws)) > 1:
            inference_data_list = [
                az.InferenceData(posterior=chain) for chain in all_chains[-1]
            ]
            if truncate:
                print(
                    f"Chains have different lengths. Truncating to shortest chain length."
                )
                return self.truncate_chains(inference_data_list)
            print(
                f"Chains have different lengths. Returning individual traces for each chain in the most recent directory."
            )
            return inference_data_list

        # Combine chains into a single InferenceData object with multiple chains
        if len(all_chains) > 1:
            combined_chains = [
                xr.concat(chain_set, dim="draw") for chain_set in zip(*all_chains)
            ]
            inference_data_list = [
                az.InferenceData(posterior=chain.expand_dims("chain"))
                for chain in combined_chains
            ]
            combined_inference_data = az.concat(inference_data_list, dim="chain")
            return combined_inference_data
        else:
            inference_data_list = [
                az.InferenceData(posterior=chain) for chain in all_chains[0]
            ]
            combined_inference_data = az.concat(inference_data_list, dim="chain")
            return combined_inference_data

    def check_convergence(self, trace):
        """Checks the convergence of the trace using R-hat values."""
        rhats = az.rhat(trace)
        return (rhats < 1.05).all()

    def posterior_predictive_checks(self, trace):
        """
        Performs posterior predictive checks (PPC) on the trace.

        Args:
            trace (az.InferenceData or pm.backends.base.MultiTrace): The trace object containing the posterior samples.
        """
        print("Posterior predictive check...")

        posterior = trace.posterior if isinstance(trace, az.InferenceData) else trace
        num_chains = len(posterior.chain)
        num_draws = len(posterior.draw)
        num_samples = 50  # Number of samples to draw for PPC

        # Generate random indices for subsampling
        chain_indices = np.random.randint(0, num_chains, size=num_samples)
        draw_indices = np.random.randint(0, num_draws, size=num_samples)

        model_outputs = []

        # Prepare SED plot if SED data is present
        if "sed" in self.data_dict:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))

        # Iterate through sampled chains and draws
        for idx in range(num_samples):
            chain_idx = chain_indices[idx]
            draw_idx = draw_indices[idx]

            # Set parameters in PHOEBE from posterior samples
            for param in posterior.data_vars:
                param_value = (
                    posterior[param].sel(chain=chain_idx, draw=draw_idx).values
                )
                if param != "sigma_lnf":
                    self.bundle.set_value(param, value=param_value)

            # Run PHOEBE computations
            if "ellcbackend" in self.bundle.computes:
                self.bundle.run_compute(compute="ellcbackend", model="latest")
            elif "fastcompute" in self.bundle.computes:
                self.bundle.run_compute(compute="fastcompute", model="latest")
            else:
                self.bundle.run_compute(model="latest")

            # SED handling
            if "sed" in self.data_dict:
                sed_obj = binarysed.SED(self.data_dict["sed"])
                teff_primary = self.bundle.get_value("teff@primary@component")
                teff_secondary = self.bundle.get_value("teff@secondary@component")
                requiv_primary = self.bundle.get_value("requiv@primary@component")
                requiv_secondary = self.bundle.get_value("requiv@secondary@component")
                logg1 = self.bundle.get_value("logg@primary@component")
                logg2 = self.bundle.get_value("logg@secondary@component")

                fig, ax = sed_obj.plot_sed_and_model(
                    teff_primary,
                    teff_secondary,
                    requiv_primary,
                    requiv_secondary,
                    logg1,
                    logg2,
                    fig=fig,
                    ax=ax,
                )

            # Extract and store LC/RV model output
            model_output = self.bundle.get_value("fluxes@model@latest")
            model_outputs.append(model_output)

        # Save the SED plot if available
        if "sed" in self.data_dict:
            fig.savefig("../ppcs/ppc_plot_sed.jpg")

        # Plot the observed vs predicted data for each dataset
        model_phases = self.bundle.to_phase(self.bundle.get_value("times@model@latest"))

        for dataset in self.data_dict:
            if dataset == "sed":
                continue

            if dataset.startswith("lc"):
                obs_data = self.data_dict[dataset]["data"]
                model_fluxes = self.bundle.get_value("fluxes@model@latest")

                # Plot observed vs predicted light curve
                fig, ax = plt.subplots(figsize=(12, 8))
