import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================================
# Config
# ============================================================
CSV_PATH = "Boston.csv"
col_names = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

num_epochs = 500
burnin_frac = 0.5

n_reps = 30
seed0 = 123

INV_JITTER = 1e-12
plot_path = "boston_logloss_cov_error_CIs.pdf"
log_x_axis = True
log_y_axis = False

# ============================================================
# Load Boston, standardize X, log target
# ============================================================
house = pd.read_csv(CSV_PATH)

X = house[col_names].to_numpy(dtype=float)
X = StandardScaler().fit_transform(X)

Y = house["MEDV"].to_numpy(dtype=float)
# Y = np.log(np.maximum(Y, 1e-12))

N, d = X.shape
print("Boston:", X.shape, Y.shape)

batch_list = [16, int(0.1*N)]

# ============================================================
# MLE under log loss (OLS)
# ============================================================
theta_hat = np.linalg.solve(X.T @ X, X.T @ Y)
mle = theta_hat.copy()

# In your script:
#   J_inverse = N * (X^T X)^{-1}
#   V = (1/N) sum r_i^2 x_i x_i^T
J_inverse = N * np.linalg.inv(X.T @ X)
J = np.linalg.inv(J_inverse)

resid = Y - X @ mle

V = np.zeros((d, d))
for i in range(N):
    xi = X[i].reshape(d, 1)
    V += (resid[i] ** 2) * (xi @ xi.T)
V /= N

# Sandwich target (scaled by 1/N like your code)
sandwich = J_inverse @ V @ J_inverse
scaled_sandwich = sandwich / N  # Sigma_star in the metric

# Ingredients for your tuning
A = (X.T @ X) / N
variance_noise = float(np.var(resid))

E_vec = (resid.T @ X) / N
E = E_vec.reshape(d, 1) @ E_vec.reshape(1, d)

def calculate_c1(cov):
    res = 0.0
    for i in range(N):
        Xi = X[i].reshape(d, 1) @ X[i].reshape(d, 1).T
        res += Xi @ cov @ Xi
    res /= N
    res -= A @ cov @ A
    return res

C1 = calculate_c1(cov=scaled_sandwich)

C2 = 0.0
for i in range(N):
    Xi = X[i].reshape(d, 1) @ X[i].reshape(d, 1).T
    C2 += (resid[i] ** 2) * Xi
C2 /= N
C2 -= E

# ============================================================
# Fast SGD path for log loss (MSE)
# ============================================================
def sgd_mse_path(theta0, X, Y, n_iters, batch_size, lr, pre_matrix, seed=None):
    """
    SGD on mean squared error: (1/2) E[(y - x^T theta)^2]
    Update: theta <- theta - pre_matrix @ (lr * grad_batch)
    where grad_batch = (Xb^T (Xb theta - Yb)) / B
    """
    if seed is not None:
        np.random.seed(seed)

    N, d = X.shape
    theta = np.array(theta0, dtype=float).copy()
    path = np.zeros((n_iters + 1, d))
    path[0] = theta

    P = np.asarray(pre_matrix, dtype=float)

    for t in range(n_iters):
        idx = np.random.randint(0, N, size=batch_size)
        Xb = X[idx]
        Yb = Y[idx]
        grad = (Xb.T @ (Xb @ theta - Yb)) / batch_size
        theta = theta - P @ (lr * grad)
        path[t + 1] = theta

    return path

def cov_rel_frob_error_from_path(path, cov_target, burnin_frac=0.5, eps=1e-12):
    path = np.asarray(path)
    T = path.shape[0]
    start = int(burnin_frac * T)
    S = path[start:]
    emp_cov = np.cov(S, rowvar=False, bias=True)
    num = np.linalg.norm(emp_cov - np.asarray(cov_target), ord="fro")
    den = np.linalg.norm(np.asarray(cov_target), ord="fro") + eps
    return float(num / den)

# ============================================================
# Build tunings (CT, LR+WS, DQ+exact (ours))
# ============================================================
def compute_Lambda_LRWS(B):
    # LR+WS: same structure as your script
    C = (A @ scaled_sandwich @ A + np.trace(A @ scaled_sandwich) * A) + (1 - d / N) * variance_noise * A
    C = C / B
    Lam = (1 / N) * (V @ J_inverse + J_inverse @ V) @ np.linalg.inv(C + (1 / N) * V + INV_JITTER * np.eye(d))
    return Lam

def compute_Lambda_DQexact(B):
    # DQ+exact (ours): your C = C1 + C2
    C = (C1 + C2) / B
    Lam = (1 / N) * (V @ J_inverse + J_inverse @ V) @ np.linalg.inv(C + (1 / N) * V + INV_JITTER * np.eye(d))
    return Lam

# ============================================================
# Run one repetition (one seed) and return rows
# ============================================================
def run_one_rep(seed):
    np.random.seed(seed)
    rows = []

    for i, B in enumerate(batch_list):
        n_iters = int(num_epochs * N / B)

        # ---- CT
        lr_ct = 2.0 * B / N
        path_ct = sgd_mse_path(mle, X, Y, n_iters, B, lr_ct, J_inverse, seed=seed + 1000 + 10*i)
        err_ct = cov_rel_frob_error_from_path(path_ct, scaled_sandwich, burnin_frac)

        # ---- LR+WS
        Lam_ws = compute_Lambda_LRWS(B)
        path_ws = sgd_mse_path(mle, X, Y, n_iters, B, 1.0, Lam_ws, seed=seed + 2000 + 10*i)
        err_ws = cov_rel_frob_error_from_path(path_ws, scaled_sandwich, burnin_frac)

        # ---- DQ+exact (ours)
        Lam_ex = compute_Lambda_DQexact(B)
        path_ex = sgd_mse_path(mle, X, Y, n_iters, B, 1.0, Lam_ex, seed=seed + 3000 + 10*i)
        err_ex = cov_rel_frob_error_from_path(path_ex, scaled_sandwich, burnin_frac)

        rows += [
            {"seed": seed, "method": "CT", "batch_size": int(B), "cov_rel_frob_error": err_ct},
            {"seed": seed, "method": "LR+WS", "batch_size": int(B), "cov_rel_frob_error": err_ws},
            {"seed": seed, "method": "DQ+exact", "batch_size": int(B), "cov_rel_frob_error": err_ex},
        ]

    return rows

# ============================================================
# CI summaries
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
# Plot
# ============================================================
def plot_cov_error_with_cis(df_ci, savepath=None):
    label_map = {"CT": "CT", "LR+WS": "LR+WS", "DQ+exact": "DQ+exact (ours)"}
    method_order = ["CT", "LR+WS", "DQ+exact"]

    plt.figure(figsize=(8, 5))

    for method in method_order:
        sub = df_ci[df_ci["method"] == method].sort_values("batch_size")
        x = sub["batch_size"].values.astype(float)
        y = sub["cov_rel_frob_error_mean"].values.astype(float)
        lo = sub["cov_rel_frob_error_lo"].values.astype(float)
        hi = sub["cov_rel_frob_error_hi"].values.astype(float)
        yerr = np.vstack([y - lo, hi - y])
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=label_map[method])

    if log_x_axis:
        plt.xscale("log")
    if log_y_axis:
        plt.yscale("log")

    plt.xlabel("batch size B")
    plt.ylabel(r"relative covariance error $\|\hat{\Sigma}-\Sigma_\star\|_F/\|\Sigma_\star\|_F$")
    plt.title("Boston (log loss): covariance error with 95% CIs")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

# ============================================================
# MAIN: run reps
# ============================================================
all_rows = []
print(f"\n=== Running {n_reps} repetitions for CIs (log loss) ===")
for r in range(n_reps):
    seed = seed0 + r
    all_rows.extend(run_one_rep(seed))

df_raw = pd.DataFrame(all_rows)
df_ci = summarize_cis(df_raw, metric="cov_rel_frob_error", group_cols=("method", "batch_size"))

print("\n=== Raw summary (mean/lo/hi/n) ===")
print(df_ci.to_string(index=False))

print("\n=== Table-ready strings ===")
table_df = make_table_ci_strings(df_ci, metric="cov_rel_frob_error", digits=3)
print(table_df.to_string(index=False))

plot_cov_error_with_cis(df_ci, savepath=plot_path)
print(f"\nSaved plot to: {plot_path}")






