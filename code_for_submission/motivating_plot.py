import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    linear_normal,
    nonlinear_exp,
    nonlinear_quadratic,
    nonlinear_dependent_noise,
)

# ============================================================
# Config
# ============================================================
dimension = 20
sample_size = 1000
num_epochs = 100
batch_list = [1, 32, 64, 128, 256, 512]

n_reps = 10
burnin_ratio = 0.5
base_seed = 123  # controls overall reproducibility
plot_name = "linear_regression_misspecify_dependent_noise_batch_size.pdf"
log_scale = False

# ============================================================
# Data generation (pick ONE)
# ============================================================

# --- Mis-specified: dependent noise
model = nonlinear_dependent_noise(
    N=sample_size, rho=0, p=dimension, sigma=0.1, penalty_para=0,
    theta_loc=0, theta_sigma=1, bias=1
)

# --- Exponential function (uncomment to use)
# model = nonlinear_exp(
#     N=sample_size, rho=0, p=dimension, sigma=0.1, penalty_para=0,
#     theta_loc=0, theta_sigma=1, bias=1
# )

# --- Quadratic function (uncomment to use)
# model = nonlinear_quadratic(
#     N=sample_size, rho=0, p=dimension, sigma=0.1, penalty_para=0,
#     theta_loc=0, theta_sigma=1, bias=1
# )

x_raw, y_raw = model.generate_data()
y_raw = y_raw.reshape(y_raw.shape[1],)

# Update sizes from actual data
sample_size = x_raw.shape[0]
dimension = x_raw.shape[1]

# ============================================================
# MLE and ingredients
# ============================================================
theta_hat = np.linalg.inv(x_raw.T @ x_raw) @ x_raw.T @ y_raw
mle = theta_hat

# J^{-1} and J
J_inverse = sample_size * np.linalg.inv(x_raw.T @ x_raw)
J = np.linalg.inv(J_inverse)

# V
V = np.zeros((dimension, dimension))
for i in range(sample_size):
    r = (y_raw[i] - x_raw[i].dot(mle))
    xi = x_raw[i].reshape(dimension, 1)
    V += (r ** 2) * (xi @ xi.T)
V /= sample_size

# Sandwich target
sandwich_matrix = J_inverse @ V @ J_inverse
scaled_sandwich_matrix = sandwich_matrix / sample_size

# Extra terms used in your DQ+exact guidance
predicted_y = x_raw @ mle
errors = y_raw - predicted_y
E_vector = (errors.T @ x_raw) / sample_size
E = E_vector.reshape(dimension, 1) @ E_vector.reshape(1, dimension)

A = (x_raw.T @ x_raw) / sample_size
variance_noise = np.var(errors)

def calculate_c1(cov):
    res = 0.0
    for i in range(sample_size):
        Xi = x_raw[i].reshape(dimension, 1) @ x_raw[i].reshape(dimension, 1).T
        res += Xi @ cov @ Xi
    res /= sample_size
    res -= A @ cov @ A
    return res

C1 = calculate_c1(cov=scaled_sandwich_matrix)

C2 = 0.0
for i in range(sample_size):
    Xi = x_raw[i].reshape(dimension, 1) @ x_raw[i].reshape(dimension, 1).T
    C2 += (errors[i] ** 2) * Xi
C2 /= sample_size
C2 -= E

# ============================================================
# Fast SGD (vectorized)
# ============================================================
def sgd_fast(
    n, batch_size, theta_0, lr_0, pre_matirx,
    fixed_lr=True, N=None, precondition=False, w1=1.0,
    fixed_point=False, regulizer=False
):
    """
    Vectorized SGD for linear regression objective.
    """
    if N is None:
        N = x_raw.shape[0]

    ret = np.zeros((n + 1, dimension))
    ret[0, :] = theta_0

    P = pre_matirx if precondition else np.eye(dimension)

    for t in range(n):
        gamma = w1 * lr_0 if fixed_lr else w1 * lr_0 / (t + 1)

        idx = np.random.randint(0, N, size=batch_size)
        Xb = x_raw[idx, :]     # (B, d)
        yb = y_raw[idx]        # (B,)

        theta = ret[t, :]

        if fixed_point:
            XtX = (Xb.T @ Xb) / batch_size
            grad = XtX @ (theta - theta_hat)
        else:
            grad = (Xb.T @ (Xb @ theta - yb)) / batch_size

        step = gamma * (P @ grad)

        if not regulizer:
            ret[t + 1, :] = theta - step
        else:
            ret[t + 1, :] = (1 - model.penalty_para * gamma / sample_size) * theta - step

    return ret

def empirical_cov(samples_path, burn_in_ratio=0.5):
    burn_in = int(len(samples_path) * burn_in_ratio)
    samples = samples_path[burn_in:, :]
    return np.cov(samples.T)

def frob_error_from_path(samples_path, cov_target, burn_in_ratio=0.5):
    cov_emp = empirical_cov(samples_path, burn_in_ratio=burn_in_ratio)
    return np.linalg.norm(cov_emp - cov_target, ord="fro")

# ============================================================
# Precompute tuning matrices per batch size
# ============================================================

# CT learning rate: 2B/N
lr_ct_list = [2 * B / sample_size for B in batch_list]

# LR+WS (discrete-quadratic + constant noise)
Lambda_dict_ws = {}
for i, B in enumerate(batch_list):
    target_cov = scaled_sandwich_matrix
    C = (A @ target_cov @ A + np.trace(A @ target_cov) * A) + (1 - dimension / sample_size) * variance_noise * A
    C /= B
    Lambda = (1 / sample_size) * (V @ J_inverse + J_inverse @ V) @ np.linalg.inv(C + (1 / sample_size) * V)
    Lambda_dict_ws[i] = Lambda

# DQ+exact (your exact guidance with C = C1 + C2)
Lambda_dict_exact = {}
for i, B in enumerate(batch_list):
    C = (C1 + C2) / B
    Lambda = (1 / sample_size) * (V @ J_inverse + J_inverse @ V) @ np.linalg.inv(C + (1 / sample_size) * V)
    Lambda_dict_exact[i] = Lambda

# ============================================================
# Run repetitions
# ============================================================
def run_one_rep(rep_seed):
    np.random.seed(rep_seed)
    initial_state = mle

    out = []
    for i, B in enumerate(batch_list):
        n_iters = num_epochs * int(sample_size / B)

        # CT
        path_ct = sgd_fast(
            n=n_iters, batch_size=B,
            theta_0=initial_state, lr_0=lr_ct_list[i],
            pre_matirx=J_inverse, fixed_lr=True, precondition=True,
            fixed_point=False, regulizer=False
        )
        err_ct = frob_error_from_path(path_ct, scaled_sandwich_matrix, burn_in_ratio)

        # LR+WS
        path_ws = sgd_fast(
            n=n_iters, batch_size=B,
            theta_0=initial_state, lr_0=1.0,
            pre_matirx=Lambda_dict_ws[i], fixed_lr=True, precondition=True,
            fixed_point=False, regulizer=False
        )
        err_ws = frob_error_from_path(path_ws, scaled_sandwich_matrix, burn_in_ratio)

        # DQ+exact
        path_exact = sgd_fast(
            n=n_iters, batch_size=B,
            theta_0=initial_state, lr_0=1.0,
            pre_matirx=Lambda_dict_exact[i], fixed_lr=True, precondition=True,
            fixed_point=False, regulizer=False
        )
        err_exact = frob_error_from_path(path_exact, scaled_sandwich_matrix, burn_in_ratio)

        out.append((B, err_ct, err_ws, err_exact))
    return out

rows = []
for r in range(n_reps):
    rep_seed = base_seed + r
    rep_out = run_one_rep(rep_seed)
    for (B, e_ct, e_ws, e_exact) in rep_out:
        rows.append({
            "rep": r,
            "batch_size": B,
            "CT": e_ct,
            "LR+WS": e_ws,
            "DQ+exact": e_exact,
        })

rep_df = pd.DataFrame(rows)

# ============================================================
# Aggregate mean + 95% CI (empirical quantiles) robustly
# ============================================================
methods = ["CT", "LR+WS", "DQ+exact(ours)"]

def summarize_group(g):
    out = {"batch_size": g.name}
    for m in methods:
        vals = g[m].to_numpy()
        out[f"{m}_mean"] = float(np.mean(vals))
        out[f"{m}_lo"] = float(np.quantile(vals, 0.025))
        out[f"{m}_hi"] = float(np.quantile(vals, 0.975))
    return pd.Series(out)

summary = rep_df.groupby("batch_size", sort=True).apply(summarize_group).reset_index(drop=True)

print("\n=== Summary (mean [2.5%, 97.5%]) over reps ===")
print(summary)

# ============================================================
# Plot with CI bands
# ============================================================
def plot_with_cis(summary_df, log_scale=False, plot_name="plot.pdf"):
    summary_df = summary_df.sort_values("batch_size")
    x = summary_df["batch_size"].to_numpy()

    plt.figure(figsize=(8, 5))

    plot_specs = [
        ("CT", "CT", "o", "-"),
        ("LR+WS", "LR+WS", "s", "--"),
        ("DQ+exact", "DQ+exact (this paper)", "^", "-."),
    ]

    for key, label, marker, ls in plot_specs:
        y  = summary_df[f"{key}_mean"].to_numpy()
        lo = summary_df[f"{key}_lo"].to_numpy()
        hi = summary_df[f"{key}_hi"].to_numpy()

        plt.plot(
            x, y,
            marker=marker,
            linestyle=ls,
            linewidth=2,
            markersize=7,
            label=label,
        )
        plt.fill_between(x, lo, hi, alpha=0.2)

    plt.xlabel("Batch Size", fontsize=16)
    plt.ylabel(r"$\|\mathcal{S}_{\star} - \hat{\mathcal{S}}\|_F$", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)

    if log_scale:
        plt.yscale("log")
        plt.ylabel(r"Log $\|\mathcal{S}_{\star} - \hat{\mathcal{S}}\|_F$", fontsize=16)

    plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()


plot_with_cis(summary, log_scale=log_scale, plot_name=plot_name)

