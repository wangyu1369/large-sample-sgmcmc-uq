#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 07:53:22 2026

@author: yu
"""

import warnings
warnings.filterwarnings("once")

import autograd.numpy as np
from autograd import grad, hessian
from autograd.scipy.stats import norm
from autograd.scipy.special import logsumexp

import pandas as pd
from sklearn.preprocessing import StandardScaler


# ------------------------------
# Load Boston Housing
# ------------------------------
def load_boston(csv_path="Boston.csv", log_target=False, add_intercept=False):
    df = pd.read_csv(csv_path)

    col_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    X = df[col_names].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    y = df['MEDV'].to_numpy(dtype=float)
    if log_target:
        y = np.log(np.maximum(y, 1e-12))

    if add_intercept:
        X = np.column_stack([X, np.ones(X.shape[0])])

    return X, y


# ------------------------------
# Stable beta-loss (log-space) with fixed Z
# ------------------------------
def beta_loss_per_sample_logspace(theta, y, x, z_i, beta=1.5, sigma=1.0):
    """
    Per-sample beta divergence loss for Gaussian working model, stable in log space.
    z_i is fixed MC sample vector for datum i.
    """
    mu = np.dot(x, theta)
    log_fy = norm.logpdf(y, mu, sigma)
    log_fz = norm.logpdf(z_i, mu, sigma)

    # - f(y)^(beta-1)/(beta-1)
    term1 = -np.exp((beta - 1.0) * log_fy) / (beta - 1.0)

    # (1/beta) E_z[ f(z)^beta ] = (1/beta) mean(exp(beta log fz))
    log_mean_fz_beta = logsumexp(beta * log_fz) - np.log(z_i.shape[0])
    term2 = np.exp(log_mean_fz_beta) / beta

    return term1 + term2


def beta_loss_batch(theta, Xb, Yb, Zb, beta=1.5, sigma=1.0):
    """
    Mean beta-loss over a minibatch, deterministic because Zb is fixed.
    Xb: (B,d), Yb: (B,), Zb: (B,n_mc)
    """
    B = Xb.shape[0]
    tot = 0.0
    for i in range(B):
        tot += beta_loss_per_sample_logspace(theta, Yb[i], Xb[i], Zb[i], beta=beta, sigma=sigma)
    return tot / B


def beta_loss_full(theta, X, Y, Z, beta=1.5, sigma=1.0):
    N = X.shape[0]
    tot = 0.0
    for i in range(N):
        tot += beta_loss_per_sample_logspace(theta, Y[i], X[i], Z[i], beta=beta, sigma=sigma)
    return tot / N


# ------------------------------
# RMSProp optimizer (for theta_hat_beta)
# ------------------------------
def rmsprop(loss_fn, theta_init, X, Y, lr=0.05, decay=0.9, eps=1e-8,
            n_iter=200, print_every=50, **loss_kwargs):
    theta = np.copy(theta_init)
    g_sq = np.zeros_like(theta)
    loss_grad = grad(lambda th, XX, YY: loss_fn(th, XX, YY, **loss_kwargs))
    hist = []
    for t in range(n_iter):
        g = loss_grad(theta, X, Y)
        g_sq = decay * g_sq + (1 - decay) * (g ** 2)
        theta = theta - lr * g / (np.sqrt(g_sq) + eps)

        L = loss_fn(theta, X, Y, **loss_kwargs)
        hist.append(float(L))
        if (print_every is not None) and (t % print_every == 0):
            print(f"RMSProp iter {t:04d} | loss = {L:.6f}")
    return theta, np.array(hist)


# ------------------------------
# J, V, C_raw for beta-loss (with fixed Z)
# ------------------------------
def empirical_fisher_beta(theta_hat, X, Y, Z, beta=1.5, sigma=1.0):
    N, d = X.shape
    g_fn = grad(lambda th, yi, xi, zi: beta_loss_per_sample_logspace(th, yi, xi, zi, beta=beta, sigma=sigma))
    V = np.zeros((d, d))
    for i in range(N):
        g = g_fn(theta_hat, Y[i], X[i], Z[i]).reshape(-1, 1)
        V += g @ g.T
    return V / N


def empirical_hessian_beta(theta_hat, X, Y, Z, beta=1.5, sigma=1.0):
    N, d = X.shape
    h_fn = hessian(lambda th, yi, xi, zi: beta_loss_per_sample_logspace(th, yi, xi, zi, beta=beta, sigma=sigma))
    Hs = np.zeros((N, d, d))
    for i in range(N):
        Hs[i] = h_fn(theta_hat, Y[i], X[i], Z[i])
    J = np.mean(Hs, axis=0)
    return Hs, J


def compute_C_raw(Hs, V, scaled_sandwich):
    """
    Same structure as your previous code:
    C_raw = E[ H_i Σ H_i ] + V - V/N
    """
    N = Hs.shape[0]
    d = Hs.shape[1]
    acc = np.zeros((d, d))
    for i in range(N):
        acc += Hs[i] @ scaled_sandwich @ Hs[i]
    acc /= N
    acc += V
    acc -= V / N
    return acc


def damped_inv(M, lam=1e-8):
    d = M.shape[0]
    return np.linalg.inv(M + lam * np.eye(d))


# ------------------------------
# SGD path under beta-loss
# ------------------------------
def sgd_beta_path(theta0, X, Y, Z, n_iters, batch_size, lr,
                 pre_matrix, beta=1.5, sigma=1.0, fixed_lr=True):
    N, d = X.shape
    theta = np.copy(theta0)
    path = np.zeros((n_iters + 1, d))
    path[0] = theta

    P = np.asarray(pre_matrix)

    # gradient of minibatch loss (deterministic because Z is fixed)
    g_fn = grad(lambda th, XX, YY, ZZ: beta_loss_batch(th, XX, YY, ZZ, beta=beta, sigma=sigma))

    for t in range(n_iters):
        gamma = lr if fixed_lr else lr / (t + 1)
        idx = np.random.randint(0, N, size=batch_size)
        Xb, Yb, Zb = X[idx], Y[idx], Z[idx]
        g = g_fn(theta, Xb, Yb, Zb)
        theta = theta - P @ (gamma * g)
        path[t + 1] = theta
    return path


# ------------------------------
# Covariance error
# ------------------------------
def frob_cov_error_from_path(path, cov_target, burnin_frac=0.5):
    T = path.shape[0]
    start = int(burnin_frac * T)
    S = path[start:]
    emp_cov = np.cov(S, rowvar=False, bias=True)
    if not np.all(np.isfinite(emp_cov)):
        return np.inf
    return float(np.linalg.norm(emp_cov - cov_target, ord="fro"))


# ------------------------------
# Main experiment
# ------------------------------
def main():
    np.random.seed(100)

    # ---- Data
    X, Y = load_boston("Boston.csv", log_target=False, add_intercept=False)
    N, d = X.shape
    print("Boston:", X.shape, Y.shape)

    # ---- Working sigma (from OLS residuals; purely for Gaussian working model scale)
    theta_ols = np.linalg.solve(X.T @ X, X.T @ Y)
    resid = Y - X @ theta_ols
    sigma_working = float(np.std(resid) + 1e-8)
    print("sigma_working:", sigma_working)

    # ---- Beta config
    beta = 1.5
    n_mc = 200

    # Freeze MC samples per datum (CRITICAL)
    Z = np.random.normal(0.0, sigma_working * 5.0, size=(N, n_mc))

    # ---- Fit theta_hat_beta
    print("\n=== Fit beta-divergence regression ===")
    theta_init = theta_ols  # good init on real data
    theta_hat_beta, _ = rmsprop(
        lambda th, XX, YY, **kw: beta_loss_full(th, XX, YY, **kw),
        theta_init, X, Y,
        lr=0.05, n_iter=200, print_every=50,
        Z=Z, beta=beta, sigma=sigma_working
    )
    print("theta_hat_beta norm:", float(np.linalg.norm(theta_hat_beta)))

    # ---- Compute J, V, sandwich target
    V = empirical_fisher_beta(theta_hat_beta, X, Y, Z, beta=beta, sigma=sigma_working)
    Hs, J = empirical_hessian_beta(theta_hat_beta, X, Y, Z, beta=beta, sigma=sigma_working)
    Jinv = damped_inv(J, lam=1e-6)

    sandwich = Jinv @ V @ Jinv
    scaled_sandwich = sandwich / N  # target covariance for constant-step regime

    # ---- C_raw for EX
    C_raw = compute_C_raw(Hs, V, scaled_sandwich)

    # ---- Build three preconditioners (CT / LS / EX)
    # CT uses J^{-1}
    # LS uses your "quadratic + well-specified-ish" C_ls structure (linear regression style)
    # EX uses C_raw / B

    A = (X.T @ X) / N
    resid_beta = Y - X @ theta_hat_beta
    variance_noise = float(np.var(resid_beta))

    b_list = [16, int(0.1 * N)]
    num_epochs = 100
    burnin_frac = 0.5

    frob = {"CT": [], "LS": [], "EX": []}

    print("\n=== Run SGD paths + report covariance errors (beta-loss) ===")
    for B in b_list:
        n_iters = int(num_epochs * N / B)

        # ----- CT
        lr_ct = 2.0 * B / N
        path_ct = sgd_beta_path(theta_hat_beta, X, Y, Z, n_iters, B, lr_ct, Jinv, beta=beta, sigma=sigma_working)
        frob["CT"].append(frob_cov_error_from_path(path_ct, scaled_sandwich, burnin_frac))

        # ----- LS
        # NOTE: this is an approximation; may be unstable for lr=1 on some real data
        C_ls = (A @ scaled_sandwich @ A + np.trace(A @ scaled_sandwich) * A) + (1 - d / N) * variance_noise * A
        C_ls = C_ls / B
        C_ls = C_ls + 1e-12 * np.eye(d)

        Lambda_ls = (1 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(C_ls + (1 / N) * V + 1e-12 * np.eye(d))
        path_ls = sgd_beta_path(theta_hat_beta, X, Y, Z, n_iters, B, 1.0, Lambda_ls, beta=beta, sigma=sigma_working)
        frob["LS"].append(frob_cov_error_from_path(path_ls, scaled_sandwich, burnin_frac))

        # ----- EX
        C_ex = (C_raw / B) + 1e-12 * np.eye(d)
        Lambda_ex = (1 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(C_ex + (1 / N) * V + 1e-12 * np.eye(d))
        path_ex = sgd_beta_path(theta_hat_beta, X, Y, Z, n_iters, B, 1.0, Lambda_ex, beta=beta, sigma=sigma_working)
        frob["EX"].append(frob_cov_error_from_path(path_ex, scaled_sandwich, burnin_frac))

        print(f"\nB={B}")
        print("  CT frob_cov:", frob["CT"][-1])
        print("  LS frob_cov:", frob["LS"][-1])
        print("  EX frob_cov:", frob["EX"][-1])

    print("\n=== Summary (beta-loss covariance error vs scaled sandwich target) ===")
    for i, B in enumerate(b_list):
        print(f"B={B:4d} | CT={frob['CT'][i]:.6g} | LS={frob['LS'][i]:.6g} | EX={frob['EX'][i]:.6g}")


if __name__ == "__main__":
    main()
