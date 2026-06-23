import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv

# -----------------------------
# Helpers
# -----------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def _logit(t, eps=1e-12):
    t = jnp.clip(t, eps, 1.0 - eps)
    return jnp.log(t) - jnp.log1p(-t)

def _normal_cdf(x):
    return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))

def _normal_ppf(u, eps=1e-12):
    u = jnp.clip(u, eps, 1.0 - eps)
    return jnp.sqrt(2.0) * erfinv(2.0 * u - 1.0)

def _logitnormal_pdf(t, mu, sigma):
    z = _logit(t)
    coeff = 1.0 / (sigma * jnp.sqrt(2.0 * jnp.pi))
    n = coeff * jnp.exp(-0.5 * ((z - mu) / sigma) ** 2)
    return n / (t * (1.0 - t))

def _logitnormal_mode_t(mu, sigma, iters=40):
    """
    Mode of logit-normal density in t-space.
    Solve in z-space:
        (2*sigmoid(z)-1) - (z-mu)/sigma^2 = 0
    """
    z = mu
    inv_s2 = 1.0 / (sigma * sigma)

    def body(z, _):
        t = _sigmoid(z)
        f = (2.0 * t - 1.0) - (z - mu) * inv_s2
        fp = 2.0 * t * (1.0 - t) - inv_s2
        z = z - f / fp
        return z, None

    z, _ = jax.lax.scan(body, z, xs=None, length=iters)
    return _sigmoid(z)

# -----------------------------
# Shifted uniform p_ts from paper
# p_ts(t; alpha) = alpha / (alpha + (1-alpha)t)^2
# -----------------------------
def shifted_uniform_pdf(t, alpha):
    return alpha / (alpha + (1.0 - alpha) * t) ** 2

# -----------------------------
# Paper-style logit-normal:
# shifting by alpha is equivalent to mu_eff = mu + log(alpha)
# -----------------------------
def shifted_logitnormal_pdf(t, alpha, mu=0.0, sigma=1.0):
    mu_eff = mu + jnp.log(alpha)
    return _logitnormal_pdf(t, mu_eff, sigma)

# -----------------------------
# Paper-style plateau-logit-normal:
# plateau AFTER constructing shifted logit-normal
# -----------------------------
def plateau_logitnormal_pdf(t, alpha, mu=0.0, sigma=1.0):
    mu_eff = mu + jnp.log(alpha)

    m = _logitnormal_mode_t(mu_eff, sigma)
    b = _logit(m)

    # Left mass under shifted logit-normal
    A = _normal_cdf((b - mu_eff) / sigma)

    # Plateau height = density at mode
    pm = _logitnormal_pdf(m, mu_eff, sigma)

    # Right plateau mass
    B = pm * (1.0 - m)

    # Normalizer
    Z = A + B

    left_pdf = _logitnormal_pdf(t, mu_eff, sigma) / Z
    right_pdf = pm / Z

    return jnp.where(t <= m, left_pdf, right_pdf)

# -----------------------------
# Sampling functions
# -----------------------------
def shifted_logit_normal_dist(rng, shape, alpha, mu=0.0, sigma=1.0):
    """
    Sample t ~ shifted-logit-normal on (0,1).
    Shift baked in: mu_eff = mu + log(alpha).
    """
    mu_eff = mu + jnp.log(alpha)
    rnd_normal = jax.random.normal(rng, shape, dtype=jnp.float32)
    return _sigmoid(rnd_normal * sigma + mu_eff).astype(jnp.float32)

def plateau_logit_normal_dist(rng, shape, alpha, mu=0.0, sigma=1.0):
    """
    Sample t ~ plateau-logit-normal(alpha, mu, sigma) on (0,1).
    Shift baked in: mu_eff = mu + log(alpha).
    Plateau: for t > mode, pdf is constant = p(mode), then renormalized.
    """
    mu_eff = mu + jnp.log(alpha)
    m = _logitnormal_mode_t(mu_eff, sigma)
    b = _logit(m)

    A = _normal_cdf((b - mu_eff) / sigma)
    pm = _logitnormal_pdf(m, mu_eff, sigma)
    B = pm * (1.0 - m)
    Z = A + B
    p_left = A / Z

    rng_u, rng_l, rng_r = jax.random.split(rng, 3)
    u = jax.random.uniform(rng_u, shape, dtype=jnp.float32)
    take_left = (u < p_left)

    u_left = jax.random.uniform(rng_l, shape, dtype=jnp.float32) * A
    z_left = mu_eff + sigma * _normal_ppf(u_left)
    t_left = _sigmoid(z_left)

    u_right = jax.random.uniform(rng_r, shape, dtype=jnp.float32)
    t_right = m + (1.0 - m) * u_right

    return jnp.where(take_left, t_left, t_right).astype(jnp.float32)

def symmetric_plateau_logit_normal_dist(rng, shape, alpha, mu=0.0, sigma=1.0):
    """
    Sample t ~ symmetric-plateau-logit-normal: reflect plateau_logit_normal around 0.5.
    If plateau_logit_normal is biased toward 1, this is biased toward 0.
    """
    return 1.0 - plateau_logit_normal_dist(rng, shape, alpha, mu, sigma)

