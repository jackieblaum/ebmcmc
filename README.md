# ebmcmc

MCMC fitting of eclipsing (and ellipsoidal) binary star systems using
[PHOEBE](http://phoebe-project.org/) forward models and the
[emcee](https://emcee.readthedocs.io/) ensemble sampler.

## Features

- Joint fitting of light curves, radial velocities, and broadband SEDs
- Learned per-dataset jitter / systematic-error parameters
- Flexible prior system (Gaia distance, RV-derived mass ratio, von Mises phase prior, etc.)
- Automatic convergence checking via integrated autocorrelation time
- Multiprocessing pool with one PHOEBE bundle per worker
- Optional eccentricity fitting
- Support for both eclipsing and non-eclipsing (ellipsoidal) binaries

## Installation

```bash
pip install -e .
```

To include optional backends (PHOEBE, ellc, binarysed):

```bash
pip install -e ".[backends]"
```

## Quick start

```python
import phoebe
from ebmcmc import EBMCMC

# 1. Build a PHOEBE bundle with your datasets attached
b = phoebe.load("my_binary.bundle")

# 2. Create the EBMCMC object
eb = EBMCMC(
    bundle=b,
    trace_dir="./traces",
    new_run_dir="run01",
    eclipsing=True,
    ecc=True,
)

# 3. Run the sampler
sampler = eb.sample(
    nwalkers=32,
    threads=16,
    prior_info={
        "gaia_dist": {"dist0": 250, "dist_lo": 230, "dist_hi": 270, "ruwe": 1.0},
    },
)

# 4. Inspect chains
chain = sampler.get_chain(flat=True, discard=2000)
```

## Parameter vector

The MCMC operates on an unconstrained parameter vector.  See the
docstring at the top of `ebmcmc/loglike.py` for the full layout,
including branching logic for RV vs photometry-only modes and the
optional eccentricity block.

| Index | Name              | Transform to physical          |
|-------|-------------------|-------------------------------|
| 0     | u_q               | q = 1 - sigmoid(u_q)         |
| 1     | log_Msum          | Msum = exp(log_Msum)          |
| 2     | log_teff1         | Teff1 = exp(log_teff1)        |
| 3     | log_tefffrac      | Teff2 = Teff1 * exp(...)      |
| 4     | log_rfrac         | R2/R1 = exp(log_rfrac)        |
| 5     | logit_rsumfrac    | (R1+R2)/a via sigmoid         |
| 6     | logit_cosi        | cos(i) via sigmoid            |
| 7+    | (mode-dependent)  | see `loglike.py` docstring    |

## `prior_info` dictionary

| Key              | Sub-keys                                    | Description                     |
|------------------|---------------------------------------------|---------------------------------|
| `t0`             | `flat`, `phi0_ref`, `kappa`, `t0_min/max`   | Phase / epoch prior             |
| `gaia_dist`      | `dist0`, `dist_lo`, `dist_hi`, `ruwe`       | Asymmetric Gaia distance prior  |
| `q_from_rv`      | `q0`, `sigma_q`                             | Gaussian mass-ratio prior       |
| `asini_from_rv`  | `asini0`, `frac_sigma`                      | Gaussian a*sin(i) prior         |
| `msum_cap`       | (scalar)                                    | Upper soft cap on total mass    |

## NERSC usage

On Perlmutter, load the PHOEBE module and submit via Slurm:

```bash
module load python
conda activate phoebe_env
srun -n 1 -c 32 python run_mcmc.py
```

Set `threads` equal to the number of CPUs requested (`-c`).

## Citation

If you use this code, please cite [Blaum et al. (in prep.)].

## License

MIT
