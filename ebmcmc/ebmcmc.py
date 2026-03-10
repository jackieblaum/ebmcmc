import phoebe
import numpy as np
import os
import emcee
import logging
from ebmcmc import loglike
from ebmcmc.loglike import sigmoid, logit, softplus_inv
from emcee.moves import StretchMove, DEMove, GaussianMove
from multiprocessing import Pool
from dustmaps.edenhofer2023 import Edenhofer2023Query
from astropy.coordinates import SkyCoord
import astropy.units as u

logger = logging.getLogger(__name__)
    
class EBMCMC:
    """
    MCMC sampler for eclipsing (and ellipsoidal) binary star systems.

    Wraps PHOEBE forward modelling with an emcee ensemble sampler.
    Supports joint fitting of light curves, radial velocities, and
    broadband SEDs, with learned per-dataset jitter parameters.

    Parameters
    ----------
    bundle : phoebe.Bundle
        Pre-configured PHOEBE bundle with datasets attached.
    trace_dir : str, optional
        Base directory for saving MCMC chains.
    sed : dict, optional
        SED data dictionary with keys ``RA``, ``DEC``, ``dist``,
        ``wavelengths``, ``fluxes``, ``flux_errs``.
    datasets : list of str, optional
        Subset of bundle datasets to fit. Defaults to all.
    eclipsing : bool
        If True, apply sin(i) prior; if False, apply non-eclipse constraints.
    ecc : bool
        If True, fit eccentricity and argument of periastron.
    prev_run_dir : str, optional
        Path to a previous run directory to resume from.
    new_run_dir : str, optional
        Name for a new run sub-directory under ``trace_dir``.
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
        teff1 = self.bundle.get_value('teff@primary@component')
        teff2 = self.bundle.get_value('teff@secondary@component')
        requiv1 = self.bundle.get_value('requiv@primary@component')
        requiv2 = self.bundle.get_value('requiv@secondary@component')
        requiv1_max = self.bundle.get_value('requiv_max@primary@component')
        requiv2_max = self.bundle.get_value('requiv_max@secondary@component')
        if requiv1 > requiv1_max:
            self.bundle.set_value('requiv@primary@component', value=requiv1_max-0.05)
        if requiv2 > requiv2_max:
            self.bundle.set_value('requiv@primary@component', value=requiv1_max-0.05) # change the primary so the secondary shifts down
            rsumfrac = self.bundle.get_value('requivsumfrac@binary@component')
            self.bundle.set_value('requivsumfrac@binary@component', value=rsumfrac - 0.03)
        # TODO: Re-enable gravity darkening and per-star LD coefficients for hot stars (T>8000K)

        if self.bundle.get_value('incl@binary@component') > 89:
            self.bundle.set_value('incl@binary@component', value=85)

        self.bundle.set_value_all("ld_mode", "lookup")
        self.bundle.set_value("eclipse_method", value="native")
        self.bundle.run_compute(compute='phoebe01', model='latest')
        pblums = self.bundle.compute_pblums(compute='phoebe01', model='latest')
        self.bundle.set_value_all("pblum_mode", "component-coupled")

        self.compute_phases = {}
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

            # Grab compute_phases directly from the dataset
            phases = self.bundle.get_value(f"compute_phases@{dataset}")

            # Safety cleanup (usually already clean, but cheap insurance)
            phases = np.mod(np.asarray(phases, dtype=float), 1.0)
            phases = np.unique(phases)
            phases.sort()

            self.compute_phases[dataset] = phases

    def initialize_logging(self):
        """Initializes logging for PHOEBE."""
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
                if self.bundle.get_value(f"{dataset}@enabled@phoebe01"):
                    logger.info('Adding dataset %s', dataset)
                    data_dict[dataset] = self.extract_light_curve_data(dataset)
            elif dataset.startswith("rv"):
                logger.info('Adding dataset %s', dataset)
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
        primary_sigmas = self.bundle.get_value(f"sigmas@{dataset}@primary")
        secondary_sigmas = self.bundle.get_value(f"sigmas@{dataset}@secondary")
        if primary_sigmas is None:
            primary_sigmas = np.ones_like(self.bundle.get_value(f"rvs@primary@{dataset}@dataset"), dtype=float)
        if secondary_sigmas is None:
            secondary_sigmas = np.ones_like(self.bundle.get_value(f"rvs@secondary@{dataset}@dataset"), dtype=float)

        return {
            "primary": self.bundle.get_value(f"rvs@primary@{dataset}@dataset"),
            "secondary": self.bundle.get_value(f"rvs@secondary@{dataset}@dataset"),
            "primary_sigmas": primary_sigmas,
            "secondary_sigmas": secondary_sigmas,
            "primary_times": primary_times,
            "secondary_times": secondary_times,
        }

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
        """
        Build the initial MCMC parameter vector from bundle values.

        Parameters
        ----------
        ecc : bool
            Whether to include eccentricity parameters.

        Returns
        -------
        list of float
            Initial parameter vector (see ``loglike`` module docstring
            for the full layout).
        """
        self.period = self.bundle.get_value("period@binary@component")
        self.t0 = self.bundle.get_value("t0_supconj@binary@component")
        m1 = self.bundle.get_value("mass@primary@component")
        m2 = self.bundle.get_value("mass@secondary@component")
        Msum_init = m1 + m2
        q_init = self.bundle.get_value("q@binary@component")
        q_init = np.clip(q_init, 1e-6, 1-1e-6)
        u_q_init = logit(1.0 - q_init)
        incl_init = self.bundle.get_value("incl@binary@component")
        rsumfrac_init = self.bundle.get_value("requivsumfrac@binary@component")
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

        logit_cosi_init = logit(cosi_init)
        logit_q_init = logit(q_init)
        log_Msum_init = np.log(Msum_init)
        log_rfrac_init = np.log(requiv2_init/requiv1_init)
        logit_rsumfrac_init = logit(rsumfrac_init)
        log_teff1_init = np.log(teff1_init)
        log_tefffrac_init = np.log(teff2_init/teff1_init)
        log_dist_init = np.log(dist_init)

        init_vals = [u_q_init, log_Msum_init, log_teff1_init,
                    log_tefffrac_init, log_rfrac_init, logit_rsumfrac_init, logit_cosi_init,
                    ]

        if not self.rvs:
            eta_sigma_lc_init  = softplus_inv(0.3*sigma_floor)
            eta_alpha_sed_init = softplus_inv(0.3*loglike.ALPHA_FLOOR)
            init_vals.append(log_dist_init)
            init_vals.append(eta_alpha_sed_init)
            init_vals.append(eta_sigma_lc_init)

        else:
            vgamma_init = self.bundle.get_value('vgamma@system')
            init_vals.append(vgamma_init)
            sigma_rv_jit_init = 1.0  # km/s
            eta_sigma_rv_init = softplus_inv(sigma_rv_jit_init)
            init_vals.append(eta_sigma_rv_init)
        if ecc:
            ecc_init = self.bundle.get_value("ecc@binary@component")
            init_vals.append(ecc_init)
            per0_init = self.bundle.get_value("per0@binary@component")
            per0_rad = np.deg2rad(per0_init)
            init_vals.append(per0_rad)

        psi_t0_init = 0.0
        init_vals.append(psi_t0_init)

        return init_vals


    def sample(self, ecc=True, nwalkers=32, nsteps=5000, threads=16, use_ellc=False,
               lc_coeff=1, rv_coeff=1, sed_coeff=1, p0=None, prior_info=None):
        """
        Run the emcee ensemble sampler with automatic convergence checking.

        Parameters
        ----------
        ecc : bool
            Fit eccentricity and argument of periastron.
        nwalkers : int
            Number of emcee walkers.
        nsteps : int
            Not used directly; convergence is checked automatically.
        threads : int
            Number of parallel worker processes.
        use_ellc : bool
            Use the ellc backend instead of PHOEBE.
        lc_coeff, rv_coeff, sed_coeff : float
            Reserved weighting coefficients (currently unused).
        p0 : array_like, optional
            Initial walker positions (nwalkers x ndim).
        prior_info : dict, optional
            Additional prior specifications passed to ``lnprior``.
            Supported keys: ``t0``, ``gaia_dist``, ``q_from_rv``,
            ``asini_from_rv``, ``msum_cap``.

        Returns
        -------
        emcee.EnsembleSampler
            The sampler object with chains accessible via ``get_chain()``.
        """

        if not use_ellc:
            phoebe.multiprocessing_set_nprocs(threads)

        initial_guess = self.get_initial_values(ecc)
        if initial_guess is None:
            raise ValueError("Initial values for parameters cannot be found.")
        
        logit_q_init = initial_guess[0]
        log_Msum_init = initial_guess[1]
        log_teff1_init = initial_guess[2]
        teff1_init = np.exp(log_teff1_init)
        log_tefffrac_init = initial_guess[3]
        log_rfrac_init = initial_guess[4]
        logit_rsumfrac_init = initial_guess[5]
        logit_cosi_init = initial_guess[6]

        log_dist_init = None

        idx = 7
        if self.rvs:
            vgamma = initial_guess[idx]
        else:
            log_dist_init = initial_guess[idx]
            alpha_sed = initial_guess[idx+1]
            sigma_lc = initial_guess[idx+2]
        if ecc and self.rvs:
            ecc_init = initial_guess[idx+1]
            per0_rad_init = initial_guess[idx+2]

        elif ecc:
            ecc_init = initial_guess[idx+3]
            per0_rad_init = initial_guess[idx+4]
        
        cosi_init = sigmoid(logit_cosi_init)
        incl_init = np.degrees(np.arccos(cosi_init))
        u_q_init = initial_guess[0]
        q_init = 1.0 - sigmoid(u_q_init)
        if q_init > 0.99:
            logit_q_scale = 0.001
        else:
            logit_q_scale = 0.01

        scales = [0.03, 0.02, 0.01, 0.01, 0.02, 0.012, 0.03]

        vgamma_scale = 2.0
        eta_sigma_rv_scale = 0.2   # log-space-ish; keep moderate to avoid huge sigma_jit proposals
        ecc_scale = 0.005
        per0_scale = 0.02

        if self.rvs:
            scales.append(vgamma_scale)
            scales.append(eta_sigma_rv_scale)
        else:
            scales.append(0.03) # distance
            scales.append(0.25)  # eta_alpha_sed
            scales.append(0.2)   # eta_sigma_lc
        if ecc:
            scales.append(ecc_scale)
            scales.append(per0_scale)

        psi_t0_scale = 0.05
        scales.append(psi_t0_scale)

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
        except (OSError, KeyError, AttributeError):
            logger.info("Starting fresh.")
            backend.reset(nwalkers, len(initial_guess))
            ndim = len(initial_guess)
            if p0 is None:
                p0 = [initial_guess + scales * np.random.randn(ndim) for _ in range(nwalkers)]
        else:
            logger.info("Sampler starting with %d steps completed.", n_steps_completed)
            p0 = backend.get_chain()[-1]

        if prior_info is None:
            prior_info = {}

        with Pool(processes=threads, initializer=loglike._pool_init, initargs=(self.data_dict, self.compute_phases, use_ellc)) as pool:
            sampler = self.run_sampler(nwalkers, ndim, backend, p0, logit_q_init, 
                                        asini_init, self.period, log_dist_init, self.t0, log_Msum_init, teff1_init,
                                        ecc, 
                                        use_ellc=use_ellc, pool=pool,
                                        lc_coeff=lc_coeff, rv_coeff=rv_coeff, 
                                        sed_coeff=sed_coeff, prior_info=prior_info)

        logger.info("Sampling completed.")

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
    
    def run_sampler(self, nwalkers, ndim, backend, p0, logit_q_init, asini_init, period_init, log_dist_init, t0, log_Msum_init, teff1_init,
                    ecc, use_ellc=False, pool=None, lc_coeff=1, rv_coeff=1, sed_coeff=1, prior_info=None):
        logger.info("Getting sampler...")

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
            logger.info('Using updated moves')
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
                                              log_Msum_init, teff1_init, self.C, ecc, self.rvs, self.eclipsing, 
                                              use_ellc, lc_coeff, rv_coeff, sed_coeff, 
                                              self.compute_phases, self.A_obs, self.sigma_A, prior_info], 
                                        pool=pool,
                                        backend=backend,
                                        moves=moves)

        logger.info("Running sampling with convergence checks...")
        logger.debug("Sampler moves: %s", sampler._moves)


        max_n = 100000  # Maximum number of steps
        thin = 1       # Keep every 10th sample to reduce autocorrelation (adjust as needed)
        burn_in = 2000  # Number of samples to discard as burn-in
        index = 0       # To track the number of autocorrelation checks
        autocorr = np.empty(max_n // (100 * thin))  # Adjusted for thinning
        old_tau = np.inf  # Previous autocorrelation time for comparison

        # Run sampling up to `max_n` steps with periodic convergence checks
        for sample in sampler.sample(p0, iterations=max_n, progress=True, thin=thin):
            # Skip initial burn-in period
            logger.debug('Sample fetched.')
    
            if sampler.iteration < burn_in:
                continue
            
            # Check convergence every 50 * thin steps
            if sampler.iteration % (50 * thin) == 0:
                # Compute the autocorrelation time
                try:
                    tau = sampler.get_autocorr_time(tol=0)
                except emcee.autocorr.AutocorrError:
                    logger.warning("Autocorrelation time could not be estimated reliably.")
                    continue

                autocorr[index] = np.mean(tau)  # Track average autocorrelation time
                index += 1

                # Check convergence criteria
                converged = np.all(tau * 50 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    logger.info("Convergence reached.")
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
        logger.info("Trace saved to %s", self.run_dir)

    def check_convergence(self, sampler):
        """Checks convergence by estimating the integrated autocorrelation time."""
        tau = sampler.get_autocorr_time(tol=0)
        return tau

    def posterior_predictive_checks(self, sampler):
        """Implements posterior predictive checks using sampled parameters."""
        pass