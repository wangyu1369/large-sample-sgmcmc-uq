#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 02:29:00 2026

@author: yu
"""

# ============================================================
# Poisson-regression SGD tuning under BOTH:
#   (1) NLL (Poisson log-likelihood)  -> CT / DQCN (T1-T2) / Exact (C1+C2 Taylor)
#   (2) beta-divergence (Poisson pmf) -> CT / DQCN-analogue / Exact-analogue
#
# Data can be generated from Negative Binomial (mis-specified) or Poisson (well-specified).
#
# NOTES (important):
# - The NLL part below follows your existing code structure (CT, T1-T2, C1+C2).
# - The beta-divergence part is added in the same “3 methods” style.
# - Computing beta J/V/Hessians at d=100 can be expensive. This script uses:
#     * subsampling of data points for J/V/H_beta estimation
#     * finite-difference gradients/Hessians for beta loss (no autograd dependency)
#   Increase `JV_SUBSAMPLE` for accuracy, decrease for speed.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import optimize
from scipy.special import gammaln

# -----------------------------
#  Data generators
# -----------------------------
class poisson_regression:
    def __init__(self, x, theta_loc=0, theta_sigma=1, bias=0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.x = x
        self.N = x.shape[0]
        self.p = x.shape[1]
        self.bias = bias
        self.true_theta = [np.random.normal(loc=theta_loc, scale=theta_sigma) + self.bias for _ in range(self.p)]

    def generate_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x = self.x
        lam = np.exp(x.dot(self.true_theta))
        y = np.random.poisson(lam=lam)
        return x, y


class negative_binomial_regression:
    def __init__(self, x, theta_loc=0, theta_sigma=1, bias=0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.x = x
        self.N = x.shape[0]
        self.p = x.shape[1]
        self.true_theta = [np.random.normal(loc=theta_loc, scale=theta_sigma) for _ in range(self.p)]
        self.bias = bias
        self.probability = 0.5

    def generate_data(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x = self.x
        # numpy NB parameterization: number of successes n, probability p
        # mean = n*(1-p)/p. Here you set n=exp(x theta + bias)
        nparam = np.exp(x.dot(self.true_theta) + self.bias)
        y = np.random.negative_binomial(n=nparam, p=self.probability)
        return x, y


# -----------------------------
#  Helpers
# -----------------------------
def stable_exp(z):
    return np.exp(np.clip(z, -50, 50))

def poisson_logpmf(k, lam):
    # log P(Y=k) for Poisson(lam)
    return -lam + k * np.log(lam + 1e-12) - gammaln(k + 1.0)

def poisson_pmf(k, lam):
    return np.exp(poisson_logpmf(k, lam))

def mgf_poisson_like(x, theta, cov, bias=0.0):
    # your mgf style: exp(x^T theta + 0.5 x^T cov x) with bias included in mean
    x = x.reshape(-1, 1)
    return np.exp(float(x.T @ theta.reshape(-1,1) + bias) + 0.5 * float(x.T @ cov @ x))


# ============================================================
# (A) NLL (Poisson log-likelihood) definitions
# ============================================================
def grad_loglik_poisson_per_sample(theta, x, y, bias=0.0):
    # gradient of log p(y|theta) for Poisson(lam=exp(x^T theta + bias))
    eta = float(x.dot(theta) + bias)
    lam = stable_exp(eta)
    return (y - lam) * x  # matches your update sign (ascent)

def hess_loglik_poisson_per_sample(theta, x, y, bias=0.0):
    # Hessian of loglik: -lam * x x^T
    eta = float(x.dot(theta) + bias)
    lam = stable_exp(eta)
    return -(lam) * np.outer(x, x)


# ============================================================
# (B) beta-divergence (Poisson pmf) definitions
#   L_i(theta) = sum_k f(k)^{1+beta}/(beta(1+beta)) - f(y)^beta/beta
#   with f = Poisson pmf.
# ============================================================
def beta_loss_poisson_per_sample(theta, x, y, beta=1.5, bias=0.0, K_cap=400):
    eta = float(x.dot(theta) + bias)
    lam = stable_exp(eta)

    # truncation
    # data-adaptive Kmax; cap for safety
    Kmax = min(K_cap, max(int(y) + 50, int(lam + 10 * np.sqrt(lam + 1e-8))))
    ks = np.arange(Kmax + 1)

    logf = poisson_logpmf(ks, lam)
    term1 = np.sum(np.exp((1.0 + beta) * logf)) / (beta * (1.0 + beta))

    logfy = poisson_logpmf(int(y), lam)
    term2 = np.exp(beta * logfy) / beta

    return term1 - term2

def beta_loss_poisson(theta, X, Y, beta=1.5, bias=0.0, K_cap=400):
    return np.mean([beta_loss_poisson_per_sample(theta, X[i], Y[i], beta=beta, bias=bias, K_cap=K_cap)
                    for i in range(X.shape[0])])


# Finite-difference grad/hess for beta-loss (keeps this self-contained)
def grad_fd_per_sample(f_per_sample, theta, x, y, eps=1e-4, **kwargs):
    d = theta.size
    g = np.zeros(d)
    for k in range(d):
        e = np.zeros(d); e[k] = 1.0
        g[k] = (f_per_sample(theta + eps*e, x, y, **kwargs) - f_per_sample(theta - eps*e, x, y, **kwargs)) / (2*eps)
    return g

def hess_fd_per_sample(f_per_sample, theta, x, y, eps=1e-4, **kwargs):
    d = theta.size
    H = np.zeros((d, d))
    for a in range(d):
        ea = np.zeros(d); ea[a] = 1.0
        for b in range(a, d):
            eb = np.zeros(d); eb[b] = 1.0
            fpp = f_per_sample(theta + eps*ea + eps*eb, x, y, **kwargs)
            fpm = f_per_sample(theta + eps*ea - eps*eb, x, y, **kwargs)
            fmp = f_per_sample(theta - eps*ea + eps*eb, x, y, **kwargs)
            fmm = f_per_sample(theta - eps*ea - eps*eb, x, y, **kwargs)
            H[a, b] = (fpp - fpm - fmp + fmm) / (4*eps*eps)
            H[b, a] = H[a, b]
    return H


# ============================================================
# SGD runner (shared)
# ============================================================
def sgd_path(n, batch_size, theta_0, lr_0, pre_matrix,
             grad_per_sample,
             X, Y,
             fixed_lr=True, w1=1.0, bias=0.0):
    """
    theta_{t+1} = theta_t + pre_matrix @ (gamma * avg_j grad_i(theta_t))
    where grad_i is typically gradient of log-likelihood (ascent),
    matching your existing sign convention.
    """
    N, d = X.shape
    ret = np.zeros((n+1, d))
    ret[0, :] = theta_0.copy()
    P = pre_matrix

    for t in range(n):
        gamma = w1 * lr_0 if fixed_lr else w1 * lr_0 / (t + 1)

        idx = np.random.randint(0, N, size=batch_size)
        g = np.zeros(d)
        th = ret[t, :]

        for j in idx:
            g += grad_per_sample(th, X[j], Y[j], bias=bias)

        g /= batch_size
        ret[t+1, :] = th + (P @ (gamma * g)).reshape(-1,)
    return ret


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    np.random.seed(0)

    # -----------------------------
    # Config
    # -----------------------------
    sample_size = 2000
    dimension_numerical = 50
    num_epochs = 50
    b_list = [16, int(0.1 * sample_size)]  # same as your example

    # beta-divergence settings
    beta = 1.5
    K_cap = 400          # truncation cap for beta-loss sum_k
    FD_EPS = 1e-4        # finite-diff step
    JV_SUBSAMPLE = 250   # <= sample_size; increase for more accurate J/V for beta

    # -----------------------------
    # Synthetic X
    # -----------------------------
    cov_x = np.ones(dimension_numerical)
    variance_list = [1, 2, 3, 4, 5]
    for i in range(dimension_numerical):
        cov_x[i] = 1.0 / (np.random.choice(variance_list) * sample_size)
    x_raw = np.random.multivariate_normal(mean=np.zeros(dimension_numerical),
                                          cov=np.diag(cov_x),
                                          size=sample_size)

    # -----------------------------
    # Generate Y (choose one)
    # -----------------------------
    # well-specified:
    # model = poisson_regression(x=x_raw, theta_loc=0, theta_sigma=1, bias=1, seed=1)
    # x_raw, y_raw = model.generate_data(seed=2)

    # mis-specified (NB):
    model = negative_binomial_regression(x=x_raw, theta_loc=0, theta_sigma=1, bias=0.1, seed=1)
    x_raw, y_raw = model.generate_data(seed=2)

    dimension = x_raw.shape[1]
    theta_star = np.asarray(model.true_theta).copy()  # "true" theta in the generator
    bias = float(model.bias)

    # -----------------------------
    # NLL "MLE" reference (your code uses true theta as mle)
    # (If you want a fitted Poisson MLE, plug in statsmodels or scipy here.)
    # -----------------------------
    mle = theta_star.copy()
    initial_state = mle.copy()

    # ============================================================
    # NLL ingredients: J, V, sandwich, plus your DQCN + Exact objects
    # ============================================================
    print("\n=== NLL ingredients (Poisson log-likelihood) ===")
    J = np.zeros((dimension, dimension))
    V = np.zeros((dimension, dimension))

    for i in range(sample_size):
        xi = x_raw[i]
        yi = y_raw[i]
        gi = grad_loglik_poisson_per_sample(mle, xi, yi, bias=bias).reshape(-1, 1)
        Hi = hess_loglik_poisson_per_sample(mle, xi, yi, bias=bias)
        # your code uses J = E[lam x x^T] (positive). Since Hess loglik is negative, flip sign:
        J += (-Hi)
        V += (gi @ gi.T)

    J /= sample_size
    V /= sample_size

    J_inverse = np.linalg.inv(J)
    sandwich_matrix = J_inverse @ V @ J_inverse
    scaled_sandwich_matrix = sandwich_matrix / sample_size

    # ---- Your DQCN object: C = T1 - T2
    def mgf(x, mean, cov):
        # match your mgf style including bias
        x = x.reshape(-1, 1)
        return np.exp(float(x.T @ mean.reshape(-1,1) + bias) + 0.5 * float(x.T @ cov @ x))

    T1 = np.zeros((dimension, dimension))
    for i in range(sample_size):
        xi = x_raw[i]
        yi = y_raw[i]
        xxT = np.outer(xi, xi)
        scalar = (yi**2
                  + mgf(2*xi, mle, scaled_sandwich_matrix)
                  - 2*yi*mgf(xi, mle, scaled_sandwich_matrix))
        T1 += scalar * xxT
    T1 /= sample_size

    T2 = np.zeros((dimension, dimension))
    for i in range(sample_size):
        xi = x_raw[i]
        yi = y_raw[i]
        for j in range(sample_size):
            xj = x_raw[j]
            yj = y_raw[j]
            scalar = (yi*yj
                      - yi*mgf(xj, mle, scaled_sandwich_matrix)
                      - yj*mgf(xi, mle, scaled_sandwich_matrix)
                      + mgf(xi + xj, mle, scaled_sandwich_matrix))
            scalar /= (sample_size**2)
            T2 += scalar * np.outer(xi, xj)

    # ---- Exact object (your Taylor block): C = C1 + C2
    predicted_y = stable_exp(x_raw @ mle + bias)
    errors = y_raw - predicted_y

    # E = E_vector E_vector^T
    E_vec = (errors.T @ x_raw) / sample_size
    E = E_vec.reshape(dimension, 1) @ E_vec.reshape(1, dimension)

    # Derivative_vector = E[ lam x x^T ]
    Derivative_vector = np.zeros((dimension, dimension))
    for i in range(sample_size):
        lam_i = predicted_y[i]
        Derivative_vector += lam_i * np.outer(x_raw[i], x_raw[i])
    Derivative_vector /= sample_size

    # C1 = E[ lam^2 (x x^T) Σ (x x^T) ] - Derivative_vector Σ Derivative_vector
    C1 = np.zeros((dimension, dimension))
    for i in range(sample_size):
        lam_i = predicted_y[i]
        xxT = np.outer(x_raw[i], x_raw[i])
        C1 += (lam_i**2) * (xxT @ scaled_sandwich_matrix @ xxT)
    C1 /= sample_size
    C1 -= (Derivative_vector @ scaled_sandwich_matrix @ Derivative_vector)

    # C2 = E[ error^2 x x^T ] - E
    C2 = np.zeros((dimension, dimension))
    for i in range(sample_size):
        C2 += (errors[i]**2) * np.outer(x_raw[i], x_raw[i])
    C2 /= sample_size
    C2 -= E

    P = scaled_sandwich_matrix @ Derivative_vector  # matches your notation

    # ============================================================
    # NLL: build 3 preconditioners for each batch size
    # ============================================================
    print("\n=== NLL preconditioners (CT / DQCN / Exact) ===")
    pre_nll_CT = {}
    pre_nll_DQCN = {}
    pre_nll_EX = {}

    for i, B in enumerate(b_list):
        # CT: J^{-1}
        pre_nll_CT[i] = J_inverse

        # DQCN: C = (T1 - T2)/B
        C = (T1 - T2) / B
        Lambda = (1.0 / sample_size) * (V @ J_inverse + J_inverse @ V) @ np.linalg.inv(C + (1.0 / sample_size) * V)
        pre_nll_DQCN[i] = Lambda

        # Exact: C = (C1 + C2)/B
        Cex = (C1 + C2) / B
        Lambda_ex = (P + P.T) @ np.linalg.inv(Cex + E + Derivative_vector @ scaled_sandwich_matrix @ Derivative_vector)
        pre_nll_EX[i] = Lambda_ex

    # ============================================================
    # Run NLL SGD paths (3 methods)
    # ============================================================
    print("\n=== Run NLL SGD paths ===")
    paths_nll = {"CT": {}, "DQCN": {}, "EX": {}}
    for i, B in enumerate(b_list):
        n_iters = num_epochs * int(sample_size / B)
        lr_ct = 2 * B / sample_size

        paths_nll["CT"][i] = sgd_path(
            n=n_iters, batch_size=B, theta_0=initial_state, lr_0=lr_ct,
            pre_matrix=pre_nll_CT[i],
            grad_per_sample=grad_loglik_poisson_per_sample,
            X=x_raw, Y=y_raw, fixed_lr=True, bias=bias
        )

        paths_nll["DQCN"][i] = sgd_path(
            n=n_iters, batch_size=B, theta_0=initial_state, lr_0=1.0,
            pre_matrix=pre_nll_DQCN[i],
            grad_per_sample=grad_loglik_poisson_per_sample,
            X=x_raw, Y=y_raw, fixed_lr=True, bias=bias
        )

        paths_nll["EX"][i] = sgd_path(
            n=n_iters, batch_size=B, theta_0=initial_state, lr_0=1.0,
            pre_matrix=pre_nll_EX[i],
            grad_per_sample=grad_loglik_poisson_per_sample,
            X=x_raw, Y=y_raw, fixed_lr=True, bias=bias
        )

    # ============================================================
    # Beta-divergence: fit theta_hat_beta (scipy)
    # ============================================================
    print("\n=== Fit beta-divergence (Poisson pmf) ===")
    theta0 = initial_state.copy()
    res = optimize.minimize(
        lambda th: beta_loss_poisson(th, x_raw, y_raw, beta=beta, bias=bias, K_cap=K_cap),
        theta0, method="L-BFGS-B", options={"maxiter": 150}
    )
    theta_hat_beta = res.x
    print("beta fit success:", res.success, "obj:", res.fun)

    # ============================================================
    # Beta ingredients: estimate J_beta, V_beta, H_beta samples (subsampled)
    # ============================================================
    print("\n=== Beta ingredients (subsampled finite-diff) ===")
    idx_jv = np.random.choice(sample_size, size=min(JV_SUBSAMPLE, sample_size), replace=False)

    V_beta = np.zeros((dimension, dimension))
    J_beta = np.zeros((dimension, dimension))
    H_beta_list = []

    for ii in idx_jv:
        xi = x_raw[ii]
        yi = y_raw[ii]

        # g_i = ∇_theta [beta-loss_i]
        gi = grad_fd_per_sample(
            beta_loss_poisson_per_sample,
            theta_hat_beta, xi, yi,
            eps=FD_EPS,
            beta=beta, bias=bias, K_cap=K_cap
        ).reshape(-1, 1)

        # H_i = ∇^2_theta [beta-loss_i]
        Hi = hess_fd_per_sample(
            beta_loss_poisson_per_sample,
            theta_hat_beta, xi, yi,
            eps=FD_EPS,
            beta=beta, bias=bias, K_cap=K_cap
        )

        V_beta += gi @ gi.T
        J_beta += Hi
        H_beta_list.append(Hi)

    V_beta /= len(idx_jv)
    J_beta /= len(idx_jv)
    Jinv_beta = np.linalg.inv(J_beta)

    sandwich_beta = Jinv_beta @ V_beta @ Jinv_beta
    scaled_sandwich_beta = sandwich_beta / sample_size

    # C_raw_beta = E[H_i Σ H_i] + V - V/N   (estimated using subsample)
    C_raw_beta = np.zeros((dimension, dimension))
    for Hi in H_beta_list:
        C_raw_beta += Hi @ scaled_sandwich_beta @ Hi
    C_raw_beta /= len(H_beta_list)
    C_raw_beta += V_beta
    C_raw_beta -= V_beta / sample_size

    # ============================================================
    # Beta: build 3 preconditioners (CT / DQCN-analogue / Exact-analogue)
    #
    # CT: J_beta^{-1} with lr 2B/N
    # DQCN-analogue: use "constant noise" approx C = (V_beta)/B
    # Exact-analogue: use C = (C_raw_beta)/B
    #
    # (This makes method #2 and #3 distinct, while keeping the same Lambda form.)
    # ============================================================
    print("\n=== Beta preconditioners (CT / DQCN-analogue / Exact-analogue) ===")
    pre_beta_CT = {}
    pre_beta_DQCN = {}
    pre_beta_EX = {}

    for i, B in enumerate(b_list):
        pre_beta_CT[i] = Jinv_beta

        # DQCN-analogue (simple constant-noise)
        C_dqcn = (V_beta / B)
        Lambda_dqcn = (1.0 / sample_size) * (V_beta @ Jinv_beta + Jinv_beta @ V_beta) @ np.linalg.inv(
            C_dqcn + (1.0 / sample_size) * V_beta
        )
        pre_beta_DQCN[i] = Lambda_dqcn

        # Exact-analogue (your discrete object)
        C_ex = (C_raw_beta / B)
        Lambda_ex = (1.0 / sample_size) * (V_beta @ Jinv_beta + Jinv_beta @ V_beta) @ np.linalg.inv(
            C_ex + (1.0 / sample_size) * V_beta
        )
        pre_beta_EX[i] = Lambda_ex

    # ============================================================
    # Run beta SGD paths (need beta gradient as "ascent" direction)
    # Your SGD expects grad-per-sample with signature (theta,x,y,bias=...)
    # We'll provide grad_beta_ascent = -∇ beta_loss_i (so we minimize beta loss).
    # ============================================================
    def grad_beta_ascent(theta, x, y, bias=0.0):
        g = grad_fd_per_sample(
            beta_loss_poisson_per_sample,
            theta, x, y,
            eps=FD_EPS,
            beta=beta, bias=bias, K_cap=K_cap
        )
        return -g  # ascent direction to minimize beta-loss

    print("\n=== Run beta SGD paths ===")
    paths_beta = {"CT": {}, "DQCN": {}, "EX": {}}
    for i, B in enumerate(b_list):
        n_iters = num_epochs * int(sample_size / B)
        lr_ct = 2 * B / sample_size

        paths_beta["CT"][i] = sgd_path(
            n=n_iters, batch_size=B, theta_0=initial_state, lr_0=lr_ct,
            pre_matrix=pre_beta_CT[i],
            grad_per_sample=grad_beta_ascent,
            X=x_raw, Y=y_raw, fixed_lr=True, bias=bias
        )

        paths_beta["DQCN"][i] = sgd_path(
            n=n_iters, batch_size=B, theta_0=initial_state, lr_0=1.0,
            pre_matrix=pre_beta_DQCN[i],
            grad_per_sample=grad_beta_ascent,
            X=x_raw, Y=y_raw, fixed_lr=True, bias=bias
        )

        paths_beta["EX"][i] = sgd_path(
            n=n_iters, batch_size=B, theta_0=initial_state, lr_0=1.0,
            pre_matrix=pre_beta_EX[i],
            grad_per_sample=grad_beta_ascent,
            X=x_raw, Y=y_raw, fixed_lr=True, bias=bias
        )

    print("\nDone.")
    print("Available paths:")
    print("  NLL :", list(paths_nll.keys()), "per batch idx 0..", len(b_list)-1)
    print("  Beta:", list(paths_beta.keys()), "per batch idx 0..", len(b_list)-1)

    # If you want: quick sanity print of final means
    for i, B in enumerate(b_list):
        print(f"\nBatch {B}:")
        for k in ["CT", "DQCN", "EX"]:
            m_nll = paths_nll[k][i].mean(axis=0)
            m_bet = paths_beta[k][i].mean(axis=0)
            print(f"  NLL-{k}  mean||.||={np.linalg.norm(m_nll):.3g}   "
                  f"Beta-{k} mean||.||={np.linalg.norm(m_bet):.3g}")


# ============================================================
# EVALUATION (Poisson simulation): param_err, calib_ks, frob_cov
# Paste AFTER paths_nll / paths_beta and target covariances exist.
# ============================================================
# import numpy as np

def quantile_calibration(samples, theta_true):
    """
    Componentwise calibration: q_j = P(Theta_j <= theta*_j).
    """
    S = np.asarray(samples)               # (T, d)
    tt = np.asarray(theta_true).reshape(1, -1)  # (1, d)
    return np.mean(S <= tt, axis=0)       # (d,)

def ks_to_uniform(qs):
    """
    KS distance between empirical CDF of qs and Unif(0,1).
    """
    qs = np.sort(np.asarray(qs))
    m = qs.size
    grid = np.arange(1, m + 1) / m
    return float(np.max(np.abs(qs - grid)))

def summarize_paths_over_batch(b_list, paths_dict, theta_target, cov_target,
                               burnin_frac=0.5):
    """
    paths_dict: dict mapping batch-index i -> path array (n_iters+1, d)
    Returns dict with arrays (len(b_list),) for:
      - param_err: ||mean - theta_target|| / ||theta_target||
      - calib_ks:  KS(q(theta*)) where q_j=P(Theta_j<=theta*_j)
      - frob_cov:  ||Cov(samples) - cov_target||_F
    """
    theta_target = np.asarray(theta_target)
    cov_target = np.asarray(cov_target)
    theta_norm = np.linalg.norm(theta_target) + 1e-12

    out = {
        "param_err": np.zeros(len(b_list)),
        "calib_ks":  np.zeros(len(b_list)),
        "frob_cov":  np.zeros(len(b_list)),
    }

    for i, B in enumerate(b_list):
        path = np.asarray(paths_dict[i])
        T = path.shape[0]
        start = int(burnin_frac * T)
        S = path[start:]  # post-burnin samples, shape (S, d)

        mean = S.mean(axis=0)
        emp_cov = np.cov(S, rowvar=False, bias=True)

        out["param_err"][i] = np.linalg.norm(mean - theta_target) / theta_norm
        qs = quantile_calibration(S, theta_target)
        out["calib_ks"][i] = ks_to_uniform(qs)
        out["frob_cov"][i] = np.linalg.norm(emp_cov - cov_target, ord="fro")

    return out

def metrics_for_reference_gaussian(theta_mean, cov_target, theta_true, n_samples=5000, seed=0):
    """
    "Reference" Gaussian sampler: draw theta ~ N(theta_mean, cov_target)
    and compute the same 3 metrics. This mirrors your linear-case
    Sandwich Gaussian / Laplace reference idea.
    """
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(np.asarray(theta_mean), np.asarray(cov_target), size=n_samples)

    mean = S.mean(axis=0)
    emp_cov = np.cov(S, rowvar=False, bias=True)

    pe = np.linalg.norm(mean - theta_true) / (np.linalg.norm(theta_true) + 1e-12)
    ks = ks_to_uniform(quantile_calibration(S, theta_true))
    frob = np.linalg.norm(emp_cov - cov_target, ord="fro")
    return pe, ks, frob

def print_wide_table(metric_name, b_list, series, floatfmt="{:.4g}"):
    """
    series: dict method_name -> dict metric_name -> array len(b_list)
    """
    methods = list(series.keys())
    header = ["B"] + methods
    print("\n" + "=" * 120)
    print(f"{metric_name} (rows=batch size, cols=methods)")
    print("=" * 120)
    print(" | ".join([f"{h:>18s}" for h in header]))
    print("-" * (21 * len(header)))

    for i, B in enumerate(b_list):
        row = [f"{int(B):>18d}"]
        for m in methods:
            row.append(f"{floatfmt.format(series[m][metric_name][i]):>18s}")
        print(" | ".join(row))


# -------------------------
# 1) Compute metrics for the 6 SGD methods
# -------------------------
burnin_frac = 0.5  # match your linear case

# NLL (log loss): CT / DQCN / EX    target covariance = scaled_sandwich_matrix
nll_CT  = summarize_paths_over_batch(b_list, paths_nll["CT"],   theta_star, scaled_sandwich_matrix, burnin_frac)
nll_DQ  = summarize_paths_over_batch(b_list, paths_nll["DQCN"], theta_star, scaled_sandwich_matrix, burnin_frac)
nll_EX  = summarize_paths_over_batch(b_list, paths_nll["EX"],   theta_star, scaled_sandwich_matrix, burnin_frac)

# beta-loss: CT / DQCN / EX         target covariance = scaled_sandwich_beta
bet_CT  = summarize_paths_over_batch(b_list, paths_beta["CT"],   theta_star, scaled_sandwich_beta, burnin_frac)
bet_DQ  = summarize_paths_over_batch(b_list, paths_beta["DQCN"], theta_star, scaled_sandwich_beta, burnin_frac)
bet_EX  = summarize_paths_over_batch(b_list, paths_beta["EX"],   theta_star, scaled_sandwich_beta, burnin_frac)

# -------------------------
# 2) Optional: add reference Gaussian baselines (like your linear table)
#    (a) "Sandwich Gaussian (beta)" : N(theta_hat_beta, scaled_sandwich_beta)
#    (b) "Laplace (log loss)"       : N(mle, scaled_sandwich_matrix)  (rough analogue)
# -------------------------
pe_sandB, ks_sandB, frob_sandB = metrics_for_reference_gaussian(theta_hat_beta, scaled_sandwich_beta, theta_star, seed=0)
pe_lapN,  ks_lapN,  frob_lapN  = metrics_for_reference_gaussian(mle,          scaled_sandwich_matrix, theta_star, seed=1)

# build series dict (method names match your legend style)
b_arr = np.asarray(b_list)
series = {
    "continuous-time (log loss)"              : dict(param_err=nll_CT["param_err"],  calib_ks=nll_CT["calib_ks"],  frob_cov=nll_CT["frob_cov"]),
    "large-sample+well-specified (log loss)"  : dict(param_err=nll_DQ["param_err"],  calib_ks=nll_DQ["calib_ks"],  frob_cov=nll_DQ["frob_cov"]),
    "exact (this paper, log loss)"            : dict(param_err=nll_EX["param_err"],  calib_ks=nll_EX["calib_ks"],  frob_cov=nll_EX["frob_cov"]),
    r"continuous-time ($\beta = 1.5$)"        : dict(param_err=bet_CT["param_err"],  calib_ks=bet_CT["calib_ks"],  frob_cov=bet_CT["frob_cov"]),
    r"large-sample+well-specified ($\beta = 1.5$)" : dict(param_err=bet_DQ["param_err"],  calib_ks=bet_DQ["calib_ks"],  frob_cov=bet_DQ["frob_cov"]),
    r"exact (this paper, ($\beta = 1.5$))"    : dict(param_err=bet_EX["param_err"],  calib_ks=bet_EX["calib_ks"],  frob_cov=bet_EX["frob_cov"]),
    r"Sandwich Gaussian ($\beta = 1.5$)"      : dict(param_err=np.full(len(b_arr), pe_sandB), calib_ks=np.full(len(b_arr), ks_sandB), frob_cov=np.full(len(b_arr), frob_sandB)),
    r"Laplace Gaussian (log loss)"            : dict(param_err=np.full(len(b_arr), pe_lapN),  calib_ks=np.full(len(b_arr), ks_lapN),  frob_cov=np.full(len(b_arr), frob_lapN)),
}

# -------------------------
# 3) Print the three tables (wide, rows=batch size, cols=methods)
# -------------------------
print_wide_table("param_err", b_list, series)
print_wide_table("calib_ks",  b_list, series)
print_wide_table("frob_cov",  b_list, series)

# If you want to exclude the two references, comment out the two "Gaussian" entries above.






