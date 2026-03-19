"""Pure math unit tests — no PHOEBE dependency."""
import numpy as np
import pytest
from ebmcmc.loglike import (
    sigmoid,
    logit,
    softplus,
    softplus_inv,
    frac,
    von_mises_logpdf,
    roche_lobe_frac,
    soft_barrier,
    interp_periodic_phase,
    _t0_from_phase_param,
    transform_params,
)


# ---- sigmoid / logit roundtrip ----

@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_sigmoid_logit_roundtrip(p):
    assert np.isclose(sigmoid(logit(p)), p, atol=1e-7)


@pytest.mark.parametrize("z", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_logit_sigmoid_roundtrip(z):
    assert np.isclose(logit(sigmoid(z)), z, atol=1e-7)


def test_sigmoid_range():
    z = np.linspace(-10, 10, 100)
    s = sigmoid(z)
    assert np.all(s > 0) and np.all(s < 1)


# ---- softplus / softplus_inv roundtrip ----

@pytest.mark.parametrize("y", [0.01, 0.5, 1.0, 5.0, 20.0])
def test_softplus_roundtrip(y):
    assert np.isclose(softplus(softplus_inv(y)), y, rtol=1e-7)


def test_softplus_positive():
    x = np.linspace(-10, 10, 100)
    assert np.all(softplus(x) > 0)


def test_softplus_large_x():
    """For large x, softplus(x) ≈ x."""
    assert np.isclose(softplus(100.0), 100.0, rtol=1e-6)


# ---- frac ----

def test_frac_basic():
    assert np.isclose(frac(1.7), 0.7)
    assert np.isclose(frac(-0.3), 0.7)
    assert np.isclose(frac(0.0), 0.0)


# ---- von_mises_logpdf ----

def test_von_mises_peak():
    """Peak of von Mises should be at phi == mu."""
    kappa = 10.0
    mu = 0.3
    phis = np.linspace(0, 1, 200)
    vals = von_mises_logpdf(phis, mu, kappa)
    assert np.isclose(phis[np.argmax(vals)], mu, atol=0.01)


def test_von_mises_kappa_zero():
    """kappa=0 gives uniform (constant 0)."""
    assert von_mises_logpdf(0.1, 0.5, 0.0) == 0.0


# ---- roche_lobe_frac ----

def test_roche_lobe_equal_mass():
    """For q=1, both Roche lobes should be equal."""
    RL1, RL2 = roche_lobe_frac(1.0)
    assert np.isclose(RL1, RL2, rtol=1e-6)


def test_roche_lobe_positive():
    for q in [0.1, 0.5, 1.0, 2.0, 10.0]:
        RL1, RL2 = roche_lobe_frac(q)
        assert RL1 > 0 and RL2 > 0
        assert RL1 < 1 and RL2 < 1


# ---- soft_barrier ----

def test_soft_barrier_inside():
    """Penalty inside bounds is much smaller than outside."""
    val_inside = soft_barrier(0.5, lower=0.0, upper=1.0, k=10.0)
    val_outside = soft_barrier(-1.0, lower=0.0, upper=1.0, k=10.0)
    # inside should be much less negative than outside
    assert val_inside > val_outside
    # softplus-based barrier always has some baseline penalty, but it should be bounded
    assert val_inside > -15.0

def test_soft_barrier_outside_lower():
    val = soft_barrier(-1.0, lower=0.0, k=10.0)
    assert val < -5.0

def test_soft_barrier_outside_upper():
    val = soft_barrier(2.0, upper=1.0, k=10.0)
    assert val < -5.0

def test_soft_barrier_no_bounds():
    assert soft_barrier(100.0) == 0.0


# ---- interp_periodic_phase ----

def test_interp_periodic_identity():
    """Interpolating at the same phases should recover original values."""
    phi = np.linspace(0, 0.99, 50)
    y = np.sin(2 * np.pi * phi)
    y_interp = interp_periodic_phase(phi, y, phi)
    assert y_interp is not None
    np.testing.assert_allclose(y_interp, y, atol=1e-10)


def test_interp_periodic_wrap():
    """Values near phase=0 and phase=1 should agree (periodicity)."""
    phi = np.linspace(0, 0.99, 100)
    y = np.cos(2 * np.pi * phi)
    y_at_0 = interp_periodic_phase(phi, y, np.array([0.001]))
    y_at_1 = interp_periodic_phase(phi, y, np.array([0.999]))
    assert y_at_0 is not None and y_at_1 is not None
    assert np.isclose(y_at_0[0], y_at_1[0], atol=0.05)


def test_interp_periodic_too_few_points():
    assert interp_periodic_phase([0.1, 0.2], [1.0, 2.0], [0.15]) is None


# ---- _t0_from_phase_param ----

def test_t0_from_phase_param_rv():
    """RV mode (no SED): psi_t0 at index 7+1+2=10 (no ecc)."""
    # 7 core + 1 eta_sigma_lc + 2 RV + 1 psi_t0 = 11
    params = np.zeros(11)
    params[10] = 0.25  # psi_t0 = 0.25
    t0, phi0 = _t0_from_phase_param(params, period=2.0, t0_ref=100.0,
                                     has_rv=True, has_sed=False, ecc_bool=False)
    assert np.isclose(phi0, 0.25)
    assert np.isclose(t0, 100.5)  # 100 + 0.25*2


def test_t0_from_phase_param_photom_ecc():
    """SED mode with ecc (no RV): psi_t0 at index 7+2+1+2=12."""
    # 7 core + 2 SED + 1 eta_sigma_lc + 2 ecc + 1 psi_t0 = 13
    params = np.zeros(13)
    params[12] = 0.5
    t0, phi0 = _t0_from_phase_param(params, period=1.0, t0_ref=50.0,
                                     has_rv=False, has_sed=True, ecc_bool=True)
    assert np.isclose(phi0, 0.5)
    assert np.isclose(t0, 50.5)


# ---- transform_params basic sanity ----

def test_transform_params_rv_mode():
    """Smoke test: transform_params returns a 12-tuple in RV mode (no SED)."""
    # 7 core + 1 eta_sigma_lc + 2 RV + 1 psi_t0 = 11
    params = np.zeros(11)
    params[0] = 0.0      # u_q -> q = 0.5
    params[1] = np.log(2.0)  # Msum = 2
    params[2] = np.log(5500)
    params[3] = np.log(0.9)  # tefffrac
    params[4] = np.log(0.8)  # rfrac
    params[5] = 0.0          # logit_rsumfrac -> ~0.5*(1-eps)
    params[6] = 0.0          # logit_cosi -> cosi = 0.5
    # params[7] = eta_sigma_lc (zero is fine)
    # params[8] = vgamma = 0
    # params[9] = eta_sigma_rv = 0
    # params[10] = psi_t0 = 0
    result = transform_params(params, period=2.0, has_rv=True, has_sed=False, ecc_bool=False)
    assert len(result) == 12
    q, Msum, teff1, teff2, r1, r2, a, incl, dist, vgamma, ecc_val, per0 = result
    assert np.isclose(q, 0.5, atol=1e-4)
    assert np.isclose(Msum, 2.0, rtol=1e-6)
    assert dist is None
    assert ecc_val is None


def test_transform_params_photom_ecc():
    """Smoke test: SED mode with eccentricity (no RV)."""
    # 7 core + 2 SED + 1 eta_sigma_lc + 2 ecc + 1 psi_t0 = 13
    params = np.zeros(13)
    params[0] = 0.0
    params[1] = np.log(2.0)
    params[2] = np.log(5500)
    params[3] = np.log(0.9)
    params[4] = np.log(0.8)
    params[5] = 0.0
    params[6] = 0.0
    params[7] = np.log(300)  # log_dist
    # params[8] = eta_alpha_sed = 0
    # params[9] = eta_sigma_lc = 0
    params[10] = 0.1         # ecc
    params[11] = 1.0         # per0_rad
    result = transform_params(params, period=2.0, has_rv=False, has_sed=True, ecc_bool=True)
    q, Msum, teff1, teff2, r1, r2, a, incl, dist, vgamma, ecc_val, per0 = result
    assert np.isclose(dist, 300.0, rtol=1e-6)
    assert vgamma is None
    assert np.isclose(ecc_val, 0.1)


def test_transform_params_all_three():
    """Smoke test: LC+SED+RV mode with eccentricity."""
    # 7 core + 2 SED + 1 eta_sigma_lc + 2 RV + 2 ecc + 1 psi_t0 = 15
    params = np.zeros(15)
    params[0] = 0.0
    params[1] = np.log(2.0)
    params[2] = np.log(5500)
    params[3] = np.log(0.9)
    params[4] = np.log(0.8)
    params[5] = 0.0
    params[6] = 0.0
    params[7] = np.log(500)   # log_dist
    # params[8] = eta_alpha_sed = 0
    # params[9] = eta_sigma_lc = 0
    params[10] = 15.0          # vgamma
    # params[11] = eta_sigma_rv = 0
    params[12] = 0.05          # ecc
    params[13] = 2.0           # per0_rad
    # params[14] = psi_t0 = 0
    result = transform_params(params, period=2.0, has_rv=True, has_sed=True, ecc_bool=True)
    q, Msum, teff1, teff2, r1, r2, a, incl, dist, vgamma, ecc_val, per0 = result
    assert np.isclose(dist, 500.0, rtol=1e-6)
    assert np.isclose(vgamma, 15.0)
    assert np.isclose(ecc_val, 0.05)
    assert np.isclose(per0, 2.0)
