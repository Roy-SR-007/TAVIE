"""
Utilities to compare the TAVIE tracked objective L(xi) against a Monte Carlo
estimate of the true ELBO for Type I and Type II SSG families.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln


def _safe_cholesky(A, jitter=1e-10, max_tries=8):
    A = 0.5 * (A + A.T)
    eye = np.eye(A.shape[0])
    cur = jitter
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(A + cur * eye)
        except np.linalg.LinAlgError:
            cur *= 10.0
    raise np.linalg.LinAlgError("Cholesky failed.")


def logpdf_mvn(x, mean, cov):
    x = np.asarray(x, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    p = mean.size
    L = _safe_cholesky(cov)
    z = np.linalg.solve(L, x - mean)
    quad = float(z @ z)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (p * np.log(2.0 * np.pi) + logdet + quad)


def logpdf_gamma_rate(x, shape, rate):
    if x <= 0:
        return -np.inf
    return shape * np.log(rate) - gammaln(shape) + (shape - 1.0) * np.log(x) - rate * x


def logpdf_normal_gamma(beta, tau2, m, V, a, b):
    return logpdf_mvn(beta, m, V / tau2) + logpdf_gamma_rate(tau2, a / 2.0, b / 2.0)


def sample_normal_gamma(m, V, a, b, n_mc, rng):
    m = np.asarray(m, dtype=float).reshape(-1)
    L = _safe_cholesky(V)
    tau2 = rng.gamma(shape=a / 2.0, scale=2.0 / b, size=n_mc)
    z = rng.normal(size=(n_mc, len(m)))
    beta = m[None, :] + (z @ L.T) / np.sqrt(tau2)[:, None]
    return beta, tau2


def sample_gaussian(m, V, n_mc, rng):
    m = np.asarray(m, dtype=float).reshape(-1)
    L = _safe_cholesky(V)
    z = rng.normal(size=(n_mc, len(m)))
    return m[None, :] + z @ L.T


# =============================
# Built-in exact log-likelihoods
# =============================

def loglik_laplace_type_I(X, y, beta, tau2):
    tau = np.sqrt(tau2)
    resid = y - X @ beta
    n = y.shape[0]
    return n * np.log(tau - np.log(2.0)) - tau * np.sum(np.abs(resid))

def loglik_student_type_I(X, y, beta, tau2, nu):
    tau = np.sqrt(tau2)
    resid = y - X @ beta
    n = y.shape[0]

    const = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
    )

    return np.sum(
        const
        + np.log(tau)
        - 0.5 * (nu + 1.0) * np.log1p(tau2 * resid**2 / nu)
    )


def loglik_binomial_type_II(X, y, beta, r):
    eta = X @ beta
    # full Binomial log pmf
    return np.sum(
        # gammaln(r + 1.0)
        # - gammaln(y + 1.0)
        # - gammaln(r - y + 1.0)
        + y * eta
        - r * np.log1p(np.exp(eta))
    )


def loglik_negbin_type_II(X, y, beta, r):
    eta = X @ beta
    # full Negative-Binomial log pmf under the parametrization
    # p = sigmoid(eta), pmf ∝ choose(y+r-1, y) p^r (1-p)^y
    return np.sum(
        # gammaln(y + r)
        # - gammaln(r)
        # - gammaln(y + 1.0)
        + r * eta
        - (r + y) * np.log1p(np.exp(eta))
    )


def _get_type_I_loglik_fn(model, family, **kwargs):
    X = model.design_matrix
    y = model.y
    family = family.lower()
    if family == 'laplace':
        return lambda beta, tau2: loglik_laplace_type_I(X, y, beta, tau2)
    if family == 'student':
        nu = kwargs.get('nu', getattr(model, 'nu', None))
        if nu is None:
            raise ValueError("Provide nu for Student's-t.")
        return lambda beta, tau2: loglik_student_type_I(X, y, beta, tau2, nu)
    if family == 'custom':
        loglik_fn = kwargs.get('loglik_fn', None)
        if loglik_fn is None:
            raise ValueError("For family='custom', provide loglik_fn(beta, tau2).")
        return lambda beta, tau2: loglik_fn(model.design_matrix, model.y, beta, tau2)
    raise ValueError("Unknown Type I family.")


def _get_type_II_loglik_fn(model, family, **kwargs):
    X = model.design_matrix
    y = model.y
    family = family.lower()
    if family == 'binomial':
        r = kwargs.get('r', getattr(model, 'r', None))
        if r is None:
            raise ValueError("Provide r for binomial.")
        return lambda beta: loglik_binomial_type_II(X, y, beta, r)
    if family == 'negbin':
        r = kwargs.get('r', getattr(model, 'r', None))
        if r is None:
            raise ValueError("Provide r for negbin.")
        return lambda beta: loglik_negbin_type_II(X, y, beta, r)
    if family == 'custom':
        loglik_fn = kwargs.get('loglik_fn', None)
        if loglik_fn is None:
            raise ValueError("For family='custom', provide loglik_fn(beta).")
        return lambda beta: loglik_fn(model.design_matrix, model.y, beta)
    raise ValueError("Unknown Type II family.")


# =============================
# True ELBO history calculators
# =============================

def compute_true_elbo_history_type_I(model, prior_params, family, alpha=None, n_mc=1000, seed=123, **kwargs):
    fv = model.fitted_values
    required = ["m_hist", "V_hist", "a_hist", "b_hist"]
    missing = [k for k in required if k not in fv]
    if missing:
        raise ValueError(
            "Model fitted_values do not contain per-iteration histories. "
            f"Missing keys: {missing}. Use the patched tavie_with_history.py backend."
        )

    m0, V0, a0, b0 = prior_params
    alpha = model.alpha if alpha is None else alpha
    rng = np.random.default_rng(seed)
    loglik_fn = _get_type_I_loglik_fn(model, family, **kwargs)

    true_elbo = []
    true_elbo_se = []

    for m, V, a, b in zip(fv['m_hist'], fv['V_hist'], fv['a_hist'], fv['b_hist']):
        beta_samps, tau2_samps = sample_normal_gamma(m, V, a, b, n_mc, rng)
        vals = np.empty(n_mc, dtype=float)
        for s in range(n_mc):
            beta_s = beta_samps[s]
            tau2_s = tau2_samps[s]
            vals[s] = (
                alpha * loglik_fn(beta_s, tau2_s)
                + logpdf_normal_gamma(beta_s, tau2_s, m0, V0, a0, b0)
                - logpdf_normal_gamma(beta_s, tau2_s, m, V, a, b)
            )
        true_elbo.append(np.mean(vals))
        true_elbo_se.append(np.std(vals, ddof=1) / np.sqrt(n_mc))

    return np.asarray(true_elbo), np.asarray(true_elbo_se)


def compute_true_elbo_history_type_II(model, prior_params, family, alpha=None, n_mc=1000, seed=123, **kwargs):
    fv = model.fitted_values
    required = ["m_hist", "V_hist"]
    missing = [k for k in required if k not in fv]
    if missing:
        raise ValueError(
            "Model fitted_values do not contain per-iteration histories. "
            f"Missing keys: {missing}. Use the patched tavie_with_history.py backend."
        )

    m0, V0 = prior_params
    alpha = model.alpha if alpha is None else alpha
    rng = np.random.default_rng(seed)
    loglik_fn = _get_type_II_loglik_fn(model, family, **kwargs)

    true_elbo = []
    true_elbo_se = []

    for m, V in zip(fv['m_hist'], fv['V_hist']):
        beta_samps = sample_gaussian(m, V, n_mc, rng)
        vals = np.empty(n_mc, dtype=float)
        for s in range(n_mc):
            beta_s = beta_samps[s]
            vals[s] = (
                alpha * loglik_fn(beta_s)
                + logpdf_mvn(beta_s, m0, V0)
                - logpdf_mvn(beta_s, m, V)
            )
        true_elbo.append(np.mean(vals))
        true_elbo_se.append(np.std(vals, ddof=1) / np.sqrt(n_mc))

    return np.asarray(true_elbo), np.asarray(true_elbo_se)


# =============================
# Plotting
# =============================

def plot_L_vs_true_elbo(model, true_elbo, true_elbo_se=None, title=""):
    L_xi = model.get_elbo()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(np.arange(len(L_xi)), L_xi, linewidth=2)
    axes[0].set_title(f"{title} $L(\\xi^{{(t)}})$")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.25)

    axes[1].plot(np.arange(len(true_elbo)), true_elbo, linewidth=2)
    if true_elbo_se is not None:
        lo = true_elbo - 1.96 * true_elbo_se
        hi = true_elbo + 1.96 * true_elbo_se
        axes[1].fill_between(np.arange(len(true_elbo)), lo, hi, alpha=0.2)
    axes[1].set_title(f"{title} True ELBO")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


# =============================
# One-shot wrappers
# =============================

def compare_type_I(model, prior_params, family, alpha=None, n_mc=1000, seed=123, title=None, **kwargs):
    true_elbo, true_elbo_se = compute_true_elbo_history_type_I(
        model=model,
        prior_params=prior_params,
        family=family,
        alpha=alpha,
        n_mc=n_mc,
        seed=seed,
        **kwargs,
    )
    #plot_L_vs_true_elbo(model, true_elbo, true_elbo_se, title=title or f"Type I: {family}")
    return true_elbo, true_elbo_se


def compare_type_II(model, prior_params, family, alpha=None, n_mc=1000, seed=123, title=None, **kwargs):
    true_elbo, true_elbo_se = compute_true_elbo_history_type_II(
        model=model,
        prior_params=prior_params,
        family=family,
        alpha=alpha,
        n_mc=n_mc,
        seed=seed,
        **kwargs,
    )
    #plot_L_vs_true_elbo(model, true_elbo, true_elbo_se, title=title or f"Type II: {family}")
    return true_elbo, true_elbo_se


# =============================
# Example execution snippets
# =============================

EXAMPLE_TYPE_I = r'''
from yourpkg.TAVIE_classes import TAVIE_loc_scale
from compare_true_elbo import compare_type_I

m0 = np.zeros(X.shape[1] + 1)   # if fit_intercept=True
V0 = np.eye(X.shape[1] + 1)
a0 = 0.05
b0 = 0.05

model = TAVIE_loc_scale(family="student", fit_intercept=True)
model.fit(X=X, y=y, prior_params=[m0, V0, a0, b0], alpha=1.0, nu=5, maxiter=500, tol=1e-9)

true_elbo, true_elbo_se = compare_type_I(
    model=model,
    prior_params=[m0, V0, a0, b0],
    family="student",
    nu=5,
    n_mc=1000,
    title="Type I Student's-t"
)
'''

EXAMPLE_TYPE_II = r'''
from yourpkg.TAVIE_classes import TAVIE_type_II
from compare_true_elbo import compare_type_II

m0 = np.zeros(X.shape[1] + 1)   # if fit_intercept=True
V0 = np.eye(X.shape[1] + 1)

model = TAVIE_type_II(family="binomial", fit_intercept=True)
model.fit(X=X, y=y, r=r, prior_params=[m0, V0], alpha=1.0, maxiter=500, tol=1e-9)

true_elbo, true_elbo_se = compare_type_II(
    model=model,
    prior_params=[m0, V0],
    family="binomial",
    r=model.r,
    n_mc=1000,
    title="Type II Binomial"
)
'''
