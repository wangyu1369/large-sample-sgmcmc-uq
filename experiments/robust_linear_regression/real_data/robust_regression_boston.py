import warnings
warnings.filterwarnings("once")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import autograd.numpy as anp
from autograd import grad, hessian
from autograd.scipy.stats import norm

from sklearn.preprocessing import StandardScaler


# ============================================================
# Config
# ============================================================

CSV_PATH = "Boston.csv"
LOG_TARGET = False
ADD_INTERCEPT = False

beta = 1.5
n_mc = 50

num_epochs = 500
burnin_frac = 0.5

# Choose batch sizes after loading N (we’ll override b later)
# b = [16, 100]

# CI config
n_reps = 30
seed0 = 123
STRICT_END_TO_END = False  # True = resample z, refit theta_hat, recompute targets each rep (slower)

# numerical stability
HESS_DAMP = 1e-6
INV_JITTER = 1e-12


# ============================================================
# Data loader
# ============================================================

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


# ============================================================
# Density-space beta-loss with FIXED z_samples
# ============================================================

def f_density(y, x, theta, sigma=1.0):
    mu = anp.dot(x, theta)
    return norm.pdf(y, mu, sigma)

def beta_loss_per_sample_fixedZ(theta, y, x, z_samples, beta=1.5, sigma=1.0, eps=1e-300):
    fy = f_density(y, x, theta, sigma)
    fy = anp.maximum(fy, eps)

    mu = anp.dot(x, theta)
    fz = norm.pdf(z_samples, mu, sigma)
    fz = anp.maximum(fz, eps)

    integral_term = anp.mean(fz ** beta)
    return - fy ** (beta - 1.0) / (beta - 1.0) + integral_term / beta

def beta_loss_fixedZ(theta, X, Y, z_samples, beta=1.5, sigma=1.0):
    N = X.shape[0]
    tot = 0.0
    for i in range(N):
        tot += beta_loss_per_sample_fixedZ(theta, Y[i], X[i], z_samples, beta=beta, sigma=sigma)
    return tot / N

def beta_loss_batch_fixedZ(theta, Xb, Yb, z_samples, beta=1.5, sigma=1.0):
    B = Xb.shape[0]
    tot = 0.0
    for i in range(B):
        tot += beta_loss_per_sample_fixedZ(theta, Yb[i], Xb[i], z_samples, beta=beta, sigma=sigma)
    return tot / B


# ============================================================
# Optimizer (RMSProp)
# ============================================================

def rmsprop(loss_fn, theta_init, X, Y, lr=0.05, decay=0.9, eps=1e-8,
            n_iter=200, print_every=50, **loss_kwargs):
    theta = anp.copy(theta_init)
    g_sq = anp.zeros_like(theta)
    loss_grad = grad(lambda th, XX, YY: loss_fn(th, XX, YY, **loss_kwargs))
    hist = []

    for t in range(n_iter):
        g = loss_grad(theta, X, Y)
        g_sq = decay * g_sq + (1 - decay) * (g ** 2)
        theta = theta - lr * g / (anp.sqrt(g_sq) + eps)

        L = loss_fn(theta, X, Y, **loss_kwargs)
        hist.append(float(L))
        if (print_every is not None) and (t % print_every == 0):
            print(f"RMSProp iter {t:04d} | loss = {float(L):.6f}")
    return np.asarray(theta), np.array(hist)


# ============================================================
# J, V, C_raw utilities
# ============================================================

def empirical_fisher_beta(theta_hat, X, Y, z_samples, beta=1.5, sigma=1.0):
    N, d = X.shape
    g_fn = grad(lambda th, yi, xi: beta_loss_per_sample_fixedZ(th, yi, xi, z_samples, beta=beta, sigma=sigma))
    V = np.zeros((d, d))
    for i in range(N):
        g = np.asarray(g_fn(theta_hat, Y[i], X[i])).reshape(-1, 1)
        V += g @ g.T
    return V / N

def empirical_hessian_beta(theta_hat, X, Y, z_samples, beta=1.5, sigma=1.0):
    N, d = X.shape
    h_fn = hessian(lambda th, yi, xi: beta_loss_per_sample_fixedZ(th, yi, xi, z_samples, beta=beta, sigma=sigma))
    Hs = np.zeros((N, d, d))
    for i in range(N):
        Hs[i] = np.asarray(h_fn(theta_hat, Y[i], X[i]))
    J = np.mean(Hs, axis=0)
    return Hs, J

def compute_C_raw(Hs, V, scaled_sandwich):
    # C_raw = E[ H_i Σ H_i ] + V - V/N
    N = Hs.shape[0]
    d = Hs.shape[1]
    acc = np.zeros((d, d))
    for i in range(N):
        acc += Hs[i] @ scaled_sandwich @ Hs[i]
    acc /= N
    acc += V
    acc -= V / N
    return acc

def damped_inv(M, lam=1e-6):
    d = M.shape[0]
    return np.linalg.inv(M + lam * np.eye(d))


# ============================================================
# SGD path under beta-loss (fixedZ minibatch loss)
# ============================================================

def sgd_beta_path(theta0, X, Y, z_samples, n_iters, batch_size, lr,
                  pre_matrix, beta=1.5, sigma=1.0, fixed_lr=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N, d = X.shape
    theta = np.array(theta0, dtype=float).copy()
    path = np.zeros((n_iters + 1, d))
    path[0] = theta

    P = np.asarray(pre_matrix, dtype=float)
    g_fn = grad(lambda th, XX, YY: beta_loss_batch_fixedZ(th, XX, YY, z_samples, beta=beta, sigma=sigma))

    for t in range(n_iters):
        gamma = lr if fixed_lr else lr / (t + 1)
        idx = np.random.randint(0, N, size=batch_size)
        Xb, Yb = X[idx], Y[idx]
        g = np.asarray(g_fn(theta, Xb, Yb))
        theta = theta - P @ (gamma * g)
        path[t + 1] = theta

    return path


# ============================================================
# Relative covariance Frobenius error from path
# ============================================================

def cov_rel_frob_error_from_path(path, cov_target, burnin_frac=0.5, eps=1e-12):
    path = np.asarray(path)
    T = path.shape[0]
    start = int(burnin_frac * T)
    S = path[start:]
    emp_cov = np.cov(S, rowvar=False, bias=True)
    if not np.all(np.isfinite(emp_cov)):
        return np.nan
    num = np.linalg.norm(emp_cov - np.asarray(cov_target), ord="fro")
    den = np.linalg.norm(np.asarray(cov_target), ord="fro") + eps
    return float(num / den)


# ============================================================
# Build tunings + run one repetition
# ============================================================

def run_one_rep(seed,
                X, Y,
                theta_hat,
                sigma_working,
                z_samples,
                beta,
                num_epochs,
                burnin_frac,
                b,
                V, Jinv, scaled_sandwich, C_raw,
                INV_JITTER=1e-12):
    np.random.seed(seed)
    N, d = X.shape
    A = (X.T @ X) / N
    resid_beta = Y - X @ theta_hat
    variance_noise = float(np.var(resid_beta))

    rows = []

    for i, B in enumerate(b):
        n_iters = int(num_epochs * N / B)

        # ---- CT
        lr_ct = 2.0 * B / N
        path_ct = sgd_beta_path(theta_hat, X, Y, z_samples, n_iters, B, lr_ct, Jinv,
                                beta=beta, sigma=sigma_working, seed=seed + 1000 + 10*i)
        err_ct = cov_rel_frob_error_from_path(path_ct, scaled_sandwich, burnin_frac)

        # ---- LS (large-sample + well-specified)
        C_ls = (A @ scaled_sandwich @ A + np.trace(A @ scaled_sandwich) * A) + (1 - d / N) * variance_noise * A
        C_ls = C_ls / B
        Lambda_ls = (1 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(C_ls + (1 / N) * V + INV_JITTER * np.eye(d))
        path_ls = sgd_beta_path(theta_hat, X, Y, z_samples, n_iters, B, 1.0, Lambda_ls,
                                beta=beta, sigma=sigma_working, seed=seed + 2000 + 10*i)
        err_ls = cov_rel_frob_error_from_path(path_ls, scaled_sandwich, burnin_frac)

        # ---- EX (exact using C_raw)
        C_ex = C_raw / B
        Lambda_ex = (1 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(C_ex + (1 / N) * V + INV_JITTER * np.eye(d))
        path_ex = sgd_beta_path(theta_hat, X, Y, z_samples, n_iters, B, 1.0, Lambda_ex,
                                beta=beta, sigma=sigma_working, seed=seed + 3000 + 10*i)
        err_ex = cov_rel_frob_error_from_path(path_ex, scaled_sandwich, burnin_frac)

        rows += [
            {"seed": seed, "method": "CT", "batch_size": int(B), "cov_rel_frob_error": err_ct},
            {"seed": seed, "method": "LS", "batch_size": int(B), "cov_rel_frob_error": err_ls},
            {"seed": seed, "method": "EX", "batch_size": int(B), "cov_rel_frob_error": err_ex},
        ]

    return rows


# ============================================================
# CI utilities
# ============================================================

def summarize_cis(df, metric, group_cols=("method", "batch_size")):
    rows = []
    for keys, sub in df.groupby(list(group_cols)):
        vals = sub[metric].values.astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            mean = lo = hi = np.nan
            n = 0
        else:
            mean = float(np.mean(vals))
            lo = float(np.quantile(vals, 0.025))
            hi = float(np.quantile(vals, 0.975))
            n = int(len(vals))

        row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        row[f"{metric}_mean"] = mean
        row[f"{metric}_lo"] = lo
        row[f"{metric}_hi"] = hi
        row[f"{metric}_n"] = n
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["batch_size", "method"])

def format_ci(mean, lo, hi, digits=3):
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"

def make_table_ci_strings(df_ci, metric, digits=3):
    out = []
    for _, r in df_ci.sort_values(["batch_size", "method"]).iterrows():
        out.append({
            "method": r["method"],
            "batch_size": int(r["batch_size"]),
            metric: format_ci(r[f"{metric}_mean"], r[f"{metric}_lo"], r[f"{metric}_hi"], digits=digits),
            f"{metric}_n": int(r[f"{metric}_n"]),
        })
    return pd.DataFrame(out).sort_values(["batch_size", "method"])


# ============================================================
# Plot: covariance error with CIs
# ============================================================

def plot_cov_error_with_cis(df_ci, b, savepath=None):
    method_order = ["CT", "LS", "EX"]
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 13})

    for method in method_order:
        sub = df_ci[df_ci["method"] == method].sort_values("batch_size")
        x = sub["batch_size"].values.astype(float)
        y = sub["cov_rel_frob_error_mean"].values.astype(float)
        lo = sub["cov_rel_frob_error_lo"].values.astype(float)
        hi = sub["cov_rel_frob_error_hi"].values.astype(float)
        yerr = np.vstack([y - lo, hi - y])
        plt.errorbar(x, y, yerr=yerr, marker='o', capsize=4, label=method)

    plt.xscale("log")
    plt.xlabel("batch size B")
    plt.ylabel("relative Frobenius covariance error")
    plt.title("Covariance error with 95% CIs")
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


# ============================================================
# Strict end-to-end repetition (optional)
# ============================================================

def build_targets_from_scratch(seed, X, Y, beta, n_mc, sigma_working):
    """
    Strict mode: resample z_samples, refit theta_hat, recompute J, V, Hs, Σ, C_raw.
    """
    np.random.seed(seed)
    N, d = X.shape

    # freeze z for this rep
    z_samples = np.random.normal(0.0, sigma_working * 5.0, size=n_mc)

    # fit theta_hat
    theta_ols = np.linalg.solve(X.T @ X, X.T @ Y)
    theta_hat, _ = rmsprop(
        lambda th, XX, YY: beta_loss_fixedZ(th, XX, YY, z_samples, beta=beta, sigma=sigma_working),
        theta_ols, X, Y,
        lr=0.05, n_iter=200, print_every=None
    )

    V = empirical_fisher_beta(theta_hat, X, Y, z_samples, beta=beta, sigma=sigma_working)
    Hs, J = empirical_hessian_beta(theta_hat, X, Y, z_samples, beta=beta, sigma=sigma_working)
    Jinv = damped_inv(J, lam=HESS_DAMP)

    sandwich = Jinv @ V @ Jinv
    scaled_sandwich = sandwich / N

    C_raw = compute_C_raw(Hs, V, scaled_sandwich)

    return theta_hat, z_samples, V, Jinv, scaled_sandwich, C_raw


# ============================================================
# MAIN
# ============================================================

# ---- Load data
np.random.seed(100)
X, Y = load_boston(CSV_PATH, log_target=LOG_TARGET, add_intercept=ADD_INTERCEPT)
N, d = X.shape
print("Boston:", X.shape, Y.shape)

# Choose b relative to N
b = [16, int(0.1 * N)]
print("Batch sizes:", b)

# ---- Working sigma from OLS residuals
theta_ols = np.linalg.solve(X.T @ X, X.T @ Y)
resid = Y - X @ theta_ols
sigma_working = float(np.std(resid) + 1e-8)
print("sigma_working:", sigma_working)

# ---- Freeze Monte Carlo samples ONCE (default mode)
np.random.seed(999)
z_samples = np.random.normal(0.0, sigma_working * 5.0, size=n_mc)

# ---- Fit beta regression ONCE (default mode)
print("\n=== Fit beta-divergence regression (density-space, fixedZ) ===")
theta_hat, loss_hist = rmsprop(
    lambda th, XX, YY: beta_loss_fixedZ(th, XX, YY, z_samples, beta=beta, sigma=sigma_working),
    theta_ols, X, Y,
    lr=0.05, n_iter=200, print_every=50
)
print("theta_hat norm:", float(np.linalg.norm(theta_hat)))

# ---- Compute targets ONCE (default mode)
V = empirical_fisher_beta(theta_hat, X, Y, z_samples, beta=beta, sigma=sigma_working)
Hs, J = empirical_hessian_beta(theta_hat, X, Y, z_samples, beta=beta, sigma=sigma_working)
Jinv = damped_inv(J, lam=HESS_DAMP)

sandwich = Jinv @ V @ Jinv
scaled_sandwich = sandwich / N
C_raw = compute_C_raw(Hs, V, scaled_sandwich)

# ---- Run repetitions
all_rows = []
print(f"\n=== Running {n_reps} repetitions for CIs (STRICT_END_TO_END={STRICT_END_TO_END}) ===")
for r in range(n_reps):
    seed = seed0 + r

    if STRICT_END_TO_END:
        th_hat_r, z_r, V_r, Jinv_r, scaled_S_r, C_raw_r = build_targets_from_scratch(
            seed, X, Y, beta, n_mc, sigma_working
        )
        rows = run_one_rep(
            seed=seed,
            X=X, Y=Y,
            theta_hat=th_hat_r,
            sigma_working=sigma_working,
            z_samples=z_r,
            beta=beta,
            num_epochs=num_epochs,
            burnin_frac=burnin_frac,
            b=b,
            V=V_r,
            Jinv=Jinv_r,
            scaled_sandwich=scaled_S_r,
            C_raw=C_raw_r,
            INV_JITTER=INV_JITTER
        )
    else:
        rows = run_one_rep(
            seed=seed,
            X=X, Y=Y,
            theta_hat=theta_hat,
            sigma_working=sigma_working,
            z_samples=z_samples,
            beta=beta,
            num_epochs=num_epochs,
            burnin_frac=burnin_frac,
            b=b,
            V=V,
            Jinv=Jinv,
            scaled_sandwich=scaled_sandwich,
            C_raw=C_raw,
            INV_JITTER=INV_JITTER
        )

    all_rows.extend(rows)

df_raw = pd.DataFrame(all_rows)
df_ci = summarize_cis(df_raw, metric="cov_rel_frob_error", group_cols=("method", "batch_size"))

print("\n=== Raw summary (mean/lo/hi/n) ===")
print(df_ci.to_string(index=False))

print("\n=== Table-ready strings ===")
table_df = make_table_ci_strings(df_ci, metric="cov_rel_frob_error", digits=3)
print(table_df.to_string(index=False))

# ---- Plot covariance error with CIs
plot_cov_error_with_cis(df_ci, b=b, savepath="boston_beta_cov_error_CIs.pdf")

import numpy as np

def cov_rel_frob_error(Sigma_method, Sigma_target, eps=1e-12):
    num = np.linalg.norm(Sigma_method - Sigma_target, ord="fro")
    den = np.linalg.norm(Sigma_target, ord="fro") + eps
    return float(num / den)

def logloss_targets_and_stdposterior(X, Y, use_sigma2_hat=True):
    """
    Returns:
      theta_hat_log
      Sigma_star_log_toggle (scaled sandwich under log loss)
      Sigma_std_log (standard posterior covariance under log loss)
    """
    N, d = X.shape
    theta_hat_log = np.linalg.solve(X.T @ X, X.T @ Y)
    resid = Y - X @ theta_hat_log

    # Jinv = N (X^T X)^{-1}
    Jinv = N * np.linalg.inv(X.T @ X)

    # V = (1/N) sum r_i^2 x_i x_i^T
    V = np.zeros((d, d))
    for i in range(N):
        xi = X[i].reshape(d, 1)
        V += (resid[i] ** 2) * (xi @ xi.T)
    V /= N

    # scaled sandwich target: (Jinv V Jinv)/N
    Sigma_star_log = (Jinv @ V @ Jinv) / N

    # standard posterior covariance under log loss
    XtX_inv = np.linalg.inv(X.T @ X)
    if use_sigma2_hat:
        sigma2_hat = float(np.mean(resid**2))
        Sigma_std_log = sigma2_hat * XtX_inv
    else:
        Sigma_std_log = XtX_inv

    return theta_hat_log, Sigma_star_log, Sigma_std_log

# ============================================================
# Report covariance errors:
#  (1) Standard posterior under log loss
#  (2) Sandwich Gaussian under beta loss
# ============================================================

import numpy as np

# ---------- utility: relative Frobenius covariance error ----------
def cov_rel_frob_error(Sigma_method, Sigma_target, eps=1e-12):
    num = np.linalg.norm(Sigma_method - Sigma_target, ord="fro")
    den = np.linalg.norm(Sigma_target, ord="fro") + eps
    return float(num / den)

# ============================================================
# (A) Standard posterior under LOG LOSS
# ============================================================

# log-loss MLE (OLS)
theta_hat_log = np.linalg.solve(X.T @ X, X.T @ Y)
resid_log = Y - X @ theta_hat_log
N, d = X.shape

# log-loss J^{-1} and V
Jinv_log = N * np.linalg.inv(X.T @ X)

V_log = np.zeros((d, d))
for i in range(N):
    xi = X[i].reshape(d, 1)
    V_log += (resid_log[i] ** 2) * (xi @ xi.T)
V_log /= N

# log-loss sandwich target
Sigma_star_log = (Jinv_log @ V_log @ Jinv_log) / N

# standard posterior covariance under log loss
# choose ONE convention (this one is statistically standard)
sigma2_hat = float(np.mean(resid_log**2))
Sigma_std_log = sigma2_hat * np.linalg.inv(X.T @ X)

# covariance error
err_std_log = cov_rel_frob_error(Sigma_std_log, Sigma_star_log)

print("CovErr(Standard posterior @ log loss):", err_std_log)


# ============================================================
# (B) Sandwich Gaussian under BETA LOSS
# ============================================================

def sandwich_gaussian_beta(Jinv, V, N):
    """
    Sandwich Gaussian covariance under beta loss:
      Sigma_SG_beta = (Jinv @ V @ Jinv) / N
    """
    return (Jinv @ V @ Jinv) / N

# uses objects already computed in your beta-loss script:
#   Jinv, V, scaled_sandwich, X
Sigma_sg_beta = sandwich_gaussian_beta(Jinv, V, X.shape[0])

# covariance error vs beta-loss sandwich target
err_sg_beta = cov_rel_frob_error(Sigma_sg_beta, scaled_sandwich)

print("CovErr(Sandwich Gaussian @ beta loss):", err_sg_beta)







