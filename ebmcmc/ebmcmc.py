import phoebe
import numpy as np
import os
import scipy.optimize
import matplotlib.pyplot as plt
import emcee
from datetime import datetime
import logging
from tqdm import tqdm
import binarysed
from ebmcmc.loglike import lnprob
from multiprocessing import Pool



class EBMCMC:
    """
    A class for performing Markov Chain Monte Carlo (MCMC) sampling on binary star systems using PHOEBE and pymc.
    """

    def __init__(
        self, bundle, trace_dir=None, sed=None, datasets=None, eclipsing=True, ecc=True
    ):
        self.bundle = bundle
        self.sed = sed
        self.min_time = 1e9
        self.max_time = 0
        self.data_dict = self.create_data_dict(datasets=datasets)
        self.eclipsing = eclipsing
        self.ecc = ecc
        self.trace_dir = trace_dir

        self.initialize_bundle()
        self.initialize_logging()
        self.set_run_dir()

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
        logging.getLogger().setLevel(logging.INFO)

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

    def get_initial_values(self, ecc):
        period_init = self.bundle.get_value("period@binary@component")
        m1 = self.bundle.get_value("mass@primary@component")
        m2 = self.bundle.get_value("mass@secondary@component")
        q_init = self.bundle.get_value("q@binary@component")
        incl_init = self.bundle.get_value("incl@binary@component")
        asini_init = self.bundle.get_value("asini@binary@component")
        requivsumfrac_init = self.bundle.get_value("requivsumfrac@binary@component")
        teffratio_init = self.bundle.get_value("teffratio@binary@component")
        teff_secondary_init = self.bundle.get_value("teff@secondary@component")
        t0_supconj_init = self.bundle.get_value('t0_supconj@binary@component')
        pblums_init = [
            self.bundle.get_value(f"pblum@primary@{dataset}@dataset")
            for dataset in self.bundle.datasets
            if dataset.startswith("lc")
        ]
        sigma_lnf_init = -10

        if q_init > 1:
            q_init = m1 / m2
            requiv_secondary_init = self.bundle.get_value("requiv@primary@component")
        else:
            requiv_secondary_init = self.bundle.get_value("requiv@secondary@component")

        if ecc:
            ecc_init = self.bundle.get_value("ecc@binary@component")
            per0_init = self.bundle.get_value("per0@binary@component")
            init_vals = [teffratio_init, incl_init, requivsumfrac_init, requiv_secondary_init, 
                q_init, t0_supconj_init, asini_init, teff_secondary_init, period_init, 
                sigma_lnf_init, ecc_init, per0_init]
            for pblum in pblums_init:
                init_vals.append(pblum)
            return init_vals
        
        init_vals = [teffratio_init, incl_init, requivsumfrac_init, requiv_secondary_init, 
                q_init, t0_supconj_init, asini_init, teff_secondary_init, period_init, 
                sigma_lnf_init]
        for pblum in pblums_init:
            init_vals.append(pblum)
        return init_vals


    def sample(self, ecc=True, nwalkers=32, nsteps=5000, threads=16):
        """Runs MCMC sampling using emcee."""

        initial_guess = self.get_initial_values(ecc)
        # print(initial_guess)
        if initial_guess is None:
            raise ValueError("Initial values for parameters cannot be found.")
        
        # Ensure the initial guess is of the right shape for emcee
        p0 = [initial_guess + 1e-4 * np.random.randn(len(initial_guess)) for _ in range(nwalkers)]
        q_init = initial_guess[4]
        period_init = initial_guess[8]
        sigma_lnf_range = [-15, -1]
        t0_range = [self.min_time, self.max_time]

        filename = '{}/mcmc.h5'.format(self.run_dir)
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, len(initial_guess))

        # Create the emcee sampler
        with Pool(processes=threads) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, len(initial_guess), lnprob, args=[self.data_dict, q_init, period_init, sigma_lnf_range, t0_range, ecc], pool=pool)

            print("Running sampling...")
            sampler.run_mcmc(p0, nsteps, progress=True)

        # Save the trace
        self.save_trace(sampler)

        return sampler
    
    def set_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.trace_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir

    def save_trace(self, sampler):

        # Save sampler chain and other attributes
        np.save(os.path.join(self.run_dir, "chain.npy"), sampler.get_chain())
        np.save(os.path.join(self.run_dir, "log_prob.npy"), sampler.get_log_prob())
        np.save(os.path.join(self.run_dir, "sampler_state.npy"), sampler.get_last_sample())
        print(f"Trace saved to {self.run_dir}")

    def check_convergence(self, sampler):
        """Checks convergence by estimating the integrated autocorrelation time."""
        tau = sampler.get_autocorr_time(tol=0)
        return tau

    def posterior_predictive_checks(self, sampler):
        """Implements posterior predictive checks using sampled parameters."""
        pass