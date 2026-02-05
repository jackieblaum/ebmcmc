import phoebe
import numpy as np
import os
import sys
import scipy.optimize
import matplotlib.pyplot as plt
import emcee
from datetime import datetime
import logging
from tqdm import tqdm
from ebmcmc import loglike
from emcee.moves import StretchMove, DEMove, KDEMove, GaussianMove
from multiprocessing import Pool
from joblib import Parallel, delayed
from dustmaps.edenhofer2023 import Edenhofer2023Query
from astropy.coordinates import SkyCoord
import astropy.units as u
    
class EBMCMC:
    """
    A class for performing Markov Chain Monte Carlo (MCMC) sampling on binary star systems using PHOEBE and pymc.
    """

    def __init__(
        self, bundle, trace_dir=None, sed=None, datasets=None, eclipsing=True, ecc=True, prev_run_dir=None, new_run_dir=None
    ):
        self.bundle = bundle
        self.sed = sed
        self.min_time = 1e9
        self.max_time = 0
        self.rvs = False
        self.compute_phases = None
        self.data_dict = self.create_data_dict(datasets=datasets)
        self.A_obs, self.sigma_A = self.estimate_ell_amp()
        self.eclipsing = eclipsing
        self.ecc = ecc
        self.trace_dir = trace_dir
        self.C = 0
        self.period = None
        self.t0 = None
        self.initialize_bundle()
        self.initialize_logging()
        self.set_run_dir(prev_run_dir, new_run_dir)

    def initialize_bundle(self):
        """Initializes PHOEBE bundle values."""
        self.bundle.set_value_all("ld_mode", "lookup")
        self.bundle.set_value("eclipse_method", value="native")
        self.bundle.run_compute(compute='phoebe01', model='latest')
        pblums = self.bundle.compute_pblums(compute='phoebe01', model='latest')
        self.bundle.set_value_all("pblum_mode", "component-coupled")
        for dataset in self.bundle.datasets:
            if not dataset.startswith('rv') and self.bundle[f'{dataset}@dataset@mask_enabled'].value:
                self.bundle.set_value(f'pblum@primary@{dataset}', pblums[f'pblum@primary@{dataset}'].value)
                times = self.bundle.get_value(f"times@{dataset}@dataset")
                if not dataset.endswith('unbinned') and len(times)==200:
                    self.compute_phases = self.bundle.to_phase(times)

            # Only LCs, only enabled
            if not dataset.startswith("lc"):
                continue
            if not self.bundle[f"{dataset}@dataset@mask_enabled"].value:
                continue
            if not dataset.endswith("unbinned"):
                continue

            self.compute_phases = {}
            # Grab compute_phases directly from the dataset
            phases = self.bundle.get_value(f"compute_phases@{dataset}")

            # Safety cleanup (usually already clean, but cheap insurance)
            phases = np.mod(np.asarray(phases, dtype=float), 1.0)
            phases = np.unique(phases)
            phases.sort()

            self.compute_phases[dataset] = phases

    def initialize_logging(self):
        """Initializes logging for PHOEBE."""
        # phoebe_logger = phoebe.logger(
        #     clevel=None, flevel="CRITICAL", filename="phoebe.log"
        # )
        phoebe_logger = phoebe.logger(clevel="WARNING", flevel="DEBUG", filename="phoebe.log")
        phoebe_logger.propagate = False
        phoebe.progressbars_off()
        logging.getLogger().setLevel(logging.WARNING)

    def create_data_dict(self, datasets=None):
        data_dict = {}
        if datasets is None:
            datasets = self.bundle.datasets

        for dataset in datasets:
            if dataset.startswith("lc"):
                data_dict[dataset] = self.extract_light_curve_data(dataset)
            elif dataset.startswith("rv"):
                data_dict[dataset] = self.extract_rv_data(dataset)
                self.rvs = True
            else:
                raise ValueError(f"Unrecognized dataset type: {dataset}")

        if self.sed:
            edenhofer = Edenhofer2023Query(integrated=True)
            RA = self.sed["RA"]
            DEC = self.sed["DEC"]
            dist = self.sed["dist"]
            coord = SkyCoord(ra=RA * u.degree,
                            dec=DEC * u.degree,
                            distance=dist * u.pc,
                            frame="icrs")
            A_base = edenhofer(coord)
            R_V = 3.1
            ebv = A_base/R_V
            self.sed["ebv"] = ebv
            data_dict["sed"] = self.sed

        return data_dict

    def extract_light_curve_data(self, dataset):
        times = self.bundle.get_value(f"times@{dataset}@dataset")
        phases = self.bundle.to_phase(times)  # ← precompute once
        self.min_time = min(self.min_time, np.min(times))
        self.max_time = max(self.max_time, np.max(times))
        return {
            "data":   self.bundle.get_value(f"fluxes@{dataset}@dataset"),
            "sigmas": self.bundle.get_value(f"sigmas@{dataset}"),
            "times":  times,
            "phases": phases,                      # ← keep here
            "passband": self.bundle.get_value(f"passband@{dataset}")  # optional, handy
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

    def logit(self, p):   
        p = np.clip(p, 1e-9, 1-1e-9)
        return np.log(p) - np.log1p(-p)

    def sigmoid(self, z): 
        return 1/(1+np.exp(-z))

    def softplus_inv(self, y):                # y > 0
        return np.log(np.expm1(y))

    def estimate_ell_amp(self):
        # use first LC with a mask enabled
        ds = next(ds for ds in self.data_dict if ds.startswith("lc"))
        y  = self.data_dict[ds]["data"]
        sig = self.data_dict[ds]["sigmas"]
        m  = np.isfinite(y) & np.isfinite(sig)
        y  = y[m]
        # robust half peak-to-peak
        p5, p95 = np.percentile(y, [5, 95])
        A_obs = 0.5*(p95 - p5)
        # uncertainty — very generous
        sigma_A = max(0.5*A_obs, 5*np.median(sig[m]))
        return max(A_obs, 1e-5), sigma_A

    def get_initial_values(self, ecc):
        self.period = self.bundle.get_value("period@binary@component")
        self.t0 = self.bundle.get_value("t0_supconj@binary@component")
        m1 = self.bundle.get_value("mass@primary@component")
        m2 = self.bundle.get_value("mass@secondary@component")
        Msum_init = m1 + m2
        q_init = self.bundle.get_value("q@binary@component")
        incl_init = self.bundle.get_value("incl@binary@component")
        # asini_init = self.bundle.get_value("asini@binary@component")
        rsumfrac_init = self.bundle.get_value("requivsumfrac@binary@component")
        # pblums_init = [
        #     self.bundle.get_value(f"pblum@primary@{dataset}@dataset")
        #     for dataset in self.bundle.datasets
        #     if dataset.startswith("lc") and self.bundle[f'{dataset}@dataset@mask_enabled'].value
        # ]
        if self.sed is not None:
            dist_init = self.sed["dist"]
        else:
            dist_init = self.bundle.get_value("distance") / 3.086e16 # m to pc

        if q_init > 1:
            q_init = m1 / m2
            requiv1_init = self.bundle.get_value("requiv@secondary@component")
            requiv2_init = self.bundle.get_value("requiv@primary@component")
            teff1_init = self.bundle.get_value("teff@secondary@component")
            teff2_init = self.bundle.get_value("teff@primary@component")
        else:
            requiv1_init = self.bundle.get_value("requiv@primary@component")
            requiv2_init = self.bundle.get_value("requiv@secondary@component")
            teff1_init = self.bundle.get_value("teff@primary@component")
            teff2_init = self.bundle.get_value("teff@secondary@component")

        if 90 < incl_init < 180:
            incl_init = 180 - incl_init

        incl_rad = np.deg2rad(incl_init)
        cosi_init = np.clip(np.cos(incl_rad), 0.0, 1.0)
        first_lc = next(ds for ds in self.data_dict if ds.startswith("lc"))
        sigma_floor = max(3e-4, 0.5*np.nanmedian(self.data_dict[first_lc]["sigmas"]))
        alpha_floor = 0.03

        logit_cosi_init = self.logit(cosi_init)
        logit_q_init = self.logit(q_init)
        log_Msum_init = np.log(Msum_init)
        log_rfrac_init = np.log(requiv2_init/requiv1_init)
        logit_rsumfrac_init = self.logit(rsumfrac_init)
        log_teff1_init = np.log(teff1_init)
        log_tefffrac_init = np.log(teff2_init/teff1_init)
        log_dist_init = np.log(dist_init)

        # self.C = log_Msum_init - 3 * log_requiv1_init
        # ridge_scale_init = 0
        # psi_t0_init = 0.0

        init_vals = [logit_q_init, log_Msum_init, log_teff1_init,
                    log_tefffrac_init, log_rfrac_init, logit_rsumfrac_init, logit_cosi_init,
                    log_dist_init]

        if not self.rvs:
            eta_sigma_lc_init  = self.softplus_inv(0.3*sigma_floor)
            eta_alpha_sed_init = self.softplus_inv(0.3*alpha_floor)
            init_vals.append(eta_alpha_sed_init)
            init_vals.append(eta_sigma_lc_init)

        else:
            vgamma_init = self.bundle.get_value('vgamma@system')
            init_vals.append(vgamma_init)
        if ecc:
            ecc_init = self.bundle.get_value("ecc@binary@component")
            init_vals.append(ecc)
            per0_init = self.bundle.get_value("per0@binary@component")
            per0_rad = np.deg2rad(per0_init)
            init_vals.append(per0_rad)
            # se = np.sqrt(max(e, 0.0))
            # xe = se * np.cos(per0_rad)
            # ye = se * np.sin(per0_rad)
            # init_vals.append(xe)
            # init_vals.append(ye)
    
        # for pblum in pblums_init:
        #     init_vals.append(pblum)

        # print("Initial Values:")
        # print("teffratio:", init_vals[0])
        # print("incl:", init_vals[1])
        # print("requivsumfrac:", init_vals[2])
        # print("requiv_secondary:", init_vals[3])
        # print("q:", init_vals[4])
        # print("t0_supconj:", init_vals[5])
        # print("asini:", init_vals[6])
        # print("teff_secondary:", init_vals[7])
        # print("period:", init_vals[8])
        # if self.rvs:
        #     print("vgamma:", init_vals[9])
        # if ecc:
        #     print("ecc:", init_vals[10])
        #     print("per0:", init_vals[11])
        # for i, pblum in enumerate(pblums_init):
        #     print(f"pblum_{i+1}:", pblum)
        return init_vals


    def sample(self, ecc=True, nwalkers=32, nsteps=5000, threads=16, use_ellc=False,
               lc_coeff=1, rv_coeff=1, sed_coeff=1, p0=None, prior_info=None):
        """Runs MCMC sampling using emcee."""

        if not use_ellc:
            phoebe.multiprocessing_set_nprocs(threads)

        initial_guess = self.get_initial_values(ecc)
        # print(initial_guess)
        if initial_guess is None:
            raise ValueError("Initial values for parameters cannot be found.")
        
        logit_q_init = initial_guess[0]
        log_Msum_init = initial_guess[1]
        log_teff1_init = initial_guess[2]
        log_tefffrac_init = initial_guess[3]
        log_rfrac_init = initial_guess[4]
        logit_rsumfrac_init = initial_guess[5]
        logit_cosi_init = initial_guess[6]
        log_dist_init = initial_guess[7]

        if self.rvs:
            vgamma = initial_guess[8]
        else:
            alpha_sed = initial_guess[8]
            sigma_lc = initial_guess[9]
        if ecc and self.rvs:
            ecc_init = initial_guess[9]
            per0_rad_init = initial_guess[10]
        elif ecc:
            ecc_init = initial_guess[10]
            per0_rad_init = initial_guess[11]
        
        # scales = [0.02, 0.2, 0.01, 0.02, 
        #         0.01, 0.0002, 0.2, 20, 0.1, 1]
        cosi_init = self.sigmoid(logit_cosi_init)
        incl_init = np.degrees(np.arccos(cosi_init))
        q_init = self.sigmoid(logit_q_init)
        if q_init > 0.99:
            logit_q_scale = 0.001
        else:
            logit_q_scale = 0.01

        log_requiv_scale = 0.02 / np.log(10)
        # scales = [log_q_scale, 
        #           0.01, 
        #           0.0005, 
        #           0.0002, 
        #           200, 
        #           200, 
        #           log_requiv_scale, 
        #           log_requiv_scale, 
        #           0.2]
        scales = [0.15, 0.1, 0.02, 0.02, 0.1, 0.1, 0.12, 0.05]
        vgamma_scale = 10
        ecc_scale = 0.01
        per0_scale = 0.01
        if self.rvs:
            scales.append(vgamma_scale)
        else:
            scales.append(0.25)
            scales.append(0.2)
        if ecc:
            scales.append(ecc_scale)
            scales.append(per0_scale)
        for _ in range(len(initial_guess) - len(scales)):
            scales.append(0.05)
        scales = np.array(scales)

        G_solar = 2942.2062175044193  # same constant you use there
        Msum_init = np.exp(log_Msum_init)
        a_init = (Msum_init * (self.period**2) * G_solar / (4*np.pi**2))**(1/3)  # in R_sun
        asini_init = a_init * np.sin(np.radians(incl_init))

        filename = '{}/mcmc.h5'.format(self.run_dir)
        backend = emcee.backends.HDFBackend(filename)

        try:
            n_steps_completed = backend.iteration
            ndim = backend.get_chain().shape[2]
        except:
            print("Starting fresh.")
            backend.reset(nwalkers, len(initial_guess))
            ndim = len(initial_guess)
            if p0 is None:
                p0 = [initial_guess + scales * np.random.randn(ndim) for _ in range(nwalkers)]
        else:
            print(f"Sampler starting with {n_steps_completed} steps completed.")
            p0 = backend.get_chain()[-1]

        # # Create the emcee sampler
        # if not use_ellc:
        #     sampler = self.run_sampler(nwalkers, ndim, backend, p0, q_init, 
        #                                asini_init, period_init, t0_init, ecc,
        #                                use_ellc=use_ellc)
        # else:

        if prior_info is None:
            prior_info = {}

        with Pool(processes=threads, initializer=loglike._pool_init, initargs=(self.data_dict, self.compute_phases, use_ellc)) as pool:
            sampler = self.run_sampler(nwalkers, ndim, backend, p0, logit_q_init, 
                                        asini_init, self.period, log_dist_init, self.t0, log_Msum_init, 
                                        ecc, 
                                        use_ellc=use_ellc, pool=pool,
                                        lc_coeff=lc_coeff, rv_coeff=rv_coeff, 
                                        sed_coeff=sed_coeff, prior_info=prior_info)

        print("Sampling completed.")
        sys.stdout.flush()

        # Save the trace
        # self.save_trace(sampler)
        return sampler

    def make_gaussian_move(self, ndim):
        # tiny local jiggle; smaller on the touchy geometry dims
        sig = np.full(ndim, 0.015, dtype=float)    # per-dim stddev
        touchy = [4, 5, 6]                          # log_rfrac, logit_rsum, logit_cosi
        for i in touchy:
            if i < ndim:
                sig[i] = 0.008
        cov = np.diag(sig**2)                       # (ndim, ndim) covariance
        return GaussianMove(cov=cov)
    
    def run_sampler(self, nwalkers, ndim, backend, p0, logit_q_init, asini_init, period_init, log_dist_init, t0, log_Msum_init,
                    ecc, use_ellc=False, pool=None, lc_coeff=1, rv_coeff=1, sed_coeff=1, prior_info=None):
        print("Getting sampler...")
        sys.stdout.flush()
        # moves = [
        #     (StretchMove(a=1.2), 0.1),
        #     (DEMove(), 0.4),
        #     (KDEMove(), 0.5)
        # ]   
        geom   = [4, 5, 6]   # log k, logit rsum, logit cosi
        sedabs = [2, 3, 7]   # log T1, dlogT, log d
        masses = [0, 1]      # logit q, log Msum
        noise  = [8, 9]      # SED frac jitter, LC jitter

        if backend.iteration < 100:
            moves = [
                (StretchMove(a=1.6),                      0.5),
                (DEMove(gamma0=0.7, nsplits=2),           0.5),
            ]
        else:
            print('Using updated moves')
            gm = self.make_gaussian_move(ndim)
            moves = [
                (StretchMove(a=1.25), 0.65),               # smaller a
                (DEMove(gamma0=0.5, nsplits=2), 0.25),    # theory-ish gamma
                (gm,                              0.10),
            ]
        
        if prior_info is None:
            prior_info = {}

        sampler = emcee.EnsembleSampler(nwalkers, 
                                        ndim, 
                                        loglike.lnprob, 
                                        args=[self.data_dict, logit_q_init, asini_init, period_init, log_dist_init, t0,
                                              log_Msum_init, self.C, ecc, self.rvs, self.eclipsing, 
                                              use_ellc, lc_coeff, rv_coeff, sed_coeff, 
                                              self.compute_phases, self.A_obs, self.sigma_A, prior_info], 
                                        pool=pool,
                                        backend=backend,
                                        moves=moves)

        print("Running sampling with convergence checks...")
        print(sampler._moves)
        print('Sampler moves: ^')
        sys.stdout.flush()

        max_n = 100000  # Maximum number of steps
        thin = 1       # Keep every 10th sample to reduce autocorrelation (adjust as needed)
        burn_in = 2000  # Number of samples to discard as burn-in
        index = 0       # To track the number of autocorrelation checks
        autocorr = np.empty(max_n // (100 * thin))  # Adjusted for thinning
        old_tau = np.inf  # Previous autocorrelation time for comparison

        # Run sampling up to `max_n` steps with periodic convergence checks
        for sample in sampler.sample(p0, iterations=max_n, progress=True, thin=thin):
            # Skip initial burn-in period
            print('Sample fetched.')
            sys.stdout.flush()
            if sampler.iteration < burn_in:
                continue
            
            # Check convergence every 50 * thin steps
            if sampler.iteration % (50 * thin) == 0:
                # Compute the autocorrelation time
                try:
                    tau = sampler.get_autocorr_time(tol=0)
                except emcee.autocorr.AutocorrError:
                    print("Autocorrelation time could not be estimated reliably.")
                    continue

                autocorr[index] = np.mean(tau)  # Track average autocorrelation time
                index += 1

                # Check convergence criteria
                converged = np.all(tau * 50 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    print("Convergence reached.")
                    break
                old_tau = tau  # Update old_tau for next comparison

        return sampler
    
    def set_run_dir(self, prev_run_dir=None, new_name=None):
        if prev_run_dir is None:
            run_dir = os.path.join(self.trace_dir, f"run_{new_name}")
            os.makedirs(run_dir, exist_ok=True)
            self.run_dir = run_dir
        else:
            self.run_dir = prev_run_dir

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