import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

from scipy.special import gammaln
from scipy.stats import norm
import statsmodels.api as sm


# =========================
# 0. Utilities
# =========================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def is_finite(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def summarize_ci(values, alpha: float = 0.05):
    """
    Percentile CI (alpha=0.05 -> 95% CI).
    Returns (mean, lo, hi, n).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    lo = np.quantile(v, alpha / 2)
    hi = np.quantile(v, 1 - alpha / 2)
    return float(np.mean(v)), float(lo), float(hi), int(v.size)


def format_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"


# =========================
# 1. Data generation / loading
# =========================

@dataclass
class SimPoissonConfig:
    N: int = 2000
    d_num: int = 40
    theta_loc: float = 0.0
    theta_sigma: float = 0.1
    bias: float = 0.5
    var_choices: Tuple[int, ...] = (1, 2, 3, 4, 5)


def sample_cov_diag_poisson(N: int, d: int, var_choices=(1, 2, 3, 4, 5)) -> np.ndarray:
    """
    Your rule: cov_x[i] = 1/(choice * N).
    """
    choices = np.random.choice(var_choices, size=d)
    cov_diag = 1.0 / (choices * N)
    return cov_diag


def generate_simulated_poisson(cfg: SimPoissonConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Returns X, y, true_theta, bias, cov_x_diag.
    """
    cov_x_diag = sample_cov_diag_poisson(cfg.N, cfg.d_num, cfg.var_choices)
    X = np.random.multivariate_normal(mean=np.zeros(cfg.d_num), cov=np.diag(cov_x_diag), size=cfg.N)

    true_theta = np.random.normal(loc=cfg.theta_loc, scale=cfg.theta_sigma, size=cfg.d_num)
    eta = X @ true_theta + cfg.bias
    mu = np.exp(eta)
    y = np.random.poisson(lam=mu)
    return X, y, true_theta, cfg.bias, cov_x_diag


# (Optional) stub for real dataset loaders
def load_german_credit(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement your German Credit preprocessing here if needed.
    """
    df = pd.read_csv(path, delimiter=";")
    y = df["Creditability"].to_numpy()
    X = df.drop("Creditability", axis=1).to_numpy()
    # Standardization etc. should go here.
    return X, y


# =========================
# 2. MLE + sandwich target
# =========================

def fit_poisson_mle_statsmodels(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Poisson GLM MLE using statsmodels.
    """
    model = sm.GLM(y, X, family=sm.families.Poisson())
    res = model.fit()
    return res.params


def compute_J_V_at_mle(X: np.ndarray, y: np.ndarray, mle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    J = E[ exp(x^T mle) x x^T ]
    V = E[ (y - exp(x^T mle))^2 x x^T ]
    Both as sample averages (divide by N).
    Vectorized version.
    """
    N, d = X.shape
    eta = X @ mle
    mu = np.exp(eta)

    # J = (1/N) * X^T diag(mu) X
    J = (X.T * mu) @ X / N

    # V = (1/N) * X^T diag((y-mu)^2) X
    r2 = (y - mu) ** 2
    V = (X.T * r2) @ X / N

    return J, V


def sandwich_covariance_scaled(J: np.ndarray, V: np.ndarray, N: int, eps: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - sandwich = J^{-1} V J^{-1}
      - scaled_sandwich = sandwich / N
    eps can stabilize inversion.
    """
    d = J.shape[0]
    if eps > 0:
        J = J + eps * np.eye(d)

    Jinv = np.linalg.inv(J)
    sandwich = Jinv @ V @ Jinv
    scaled = sandwich / N
    return sandwich, scaled


# =========================
# 3. Preconditioners (your CT / improved / taylor)
# =========================

def precond_CT(Jinv: np.ndarray) -> np.ndarray:
    """
    Your baseline preconditioner matrix (full inverse of J).
    """
    return Jinv


def precond_improved_quadratic_constant_noise(
    J: np.ndarray,
    V: np.ndarray,
    Jinv: np.ndarray,
    N: int,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Your improved Lambda:
      Lambda = (1/N) (V J^{-1} + J^{-1} V) (C + (1/N) V)^{-1}
    with C = J in your code.
    """
    d = J.shape[0]
    C = J
    M = C + (1.0 / N) * V + eps * np.eye(d)
    Lambda = (1.0 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(M)
    return Lambda


def compute_taylor_quantities(
    X: np.ndarray,
    y: np.ndarray,
    mle: np.ndarray,
    scaled_sandwich: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes (C1, C2, P) for your first-order Taylor preconditioner.

    - Derivative_vector = E[ exp(x^T mle) x x^T ] = J
    - errors = y - exp(x^T mle)
    - E = (E[errors x]) (E[errors x])^T
    - C2 = E[ errors^2 x x^T ] - E
    - C1 = E[ exp(2 x^T mle) (x x^T) cov (x x^T) ] - J cov J
    - P = cov * J
    """
    N, d = X.shape
    eta = X @ mle
    mu = np.exp(eta)

    # J = E[mu x x^T]
    J = (X.T * mu) @ X / N

    errors = y - mu
    Ex = (errors @ X) / N
    E = Ex.reshape(d, 1) @ Ex.reshape(1, d)

    C2 = (X.T * (errors ** 2)) @ X / N - E

    # C1 term: loop (kept, because it's 4th order in X unless you exploit structure)
    C1 = np.zeros((d, d))
    for i in range(N):
        xxT = X[i].reshape(d, 1) @ X[i].reshape(1, d)
        C1 += (mu[i] ** 2) * (xxT @ scaled_sandwich @ xxT)
    C1 /= N
    C1 -= J @ scaled_sandwich @ J

    P = scaled_sandwich @ J
    return C1, C2, P


def precond_taylor(
    C1: np.ndarray,
    C2: np.ndarray,
    P: np.ndarray,
    J: np.ndarray,
    scaled_sandwich: np.ndarray,
    batch_size: int
) -> np.ndarray:
    """
    Your Taylor Lambda:
      C = (C1 + C2)/B
      Lambda = (P+P^T) ( C + J cov J )^{-1}
    Here J = Derivative_vector, cov = scaled_sandwich.
    """
    C = (C1 + C2) / float(batch_size)
    denom = C + J @ scaled_sandwich @ J
    Lambda = (P + P.T) @ np.linalg.inv(denom)
    return Lambda


# =========================
# 4. SGD sampler
# =========================

def sgd_poisson(
    n_steps: int,
    batch_size: int,
    theta0: np.ndarray,
    lr: float,
    pre_mat: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    mle: np.ndarray,
    fixed_lr: bool = True,
    approximate_loss: bool = False,
) -> np.ndarray:
    """
    Your SGD-like update, organized.

    If approximate_loss=False: uses Poisson gradient estimator from minibatch.
    If approximate_loss=True: uses your quadratic approximation at mle.
    """
    N, d = X.shape
    path = np.zeros((n_steps + 1, d))
    path[0] = theta0

    for t in range(n_steps):
        gamma = lr if fixed_lr else lr / (t + 1)

        idx = np.random.randint(0, N, size=batch_size)
        Xb = X[idx]
        yb = y[idx]

        theta = path[t]

        if not approximate_loss:
            # minibatch gradient of average log-likelihood (your sign convention)
            eta = Xb @ theta
            mu = np.exp(eta)
            grad = ((yb - mu)[:, None] * Xb).mean(axis=0)
        else:
            # quadratic approx around mle: E[exp(x^T mle) x x^T] (theta - mle)
            # Note: this matches your earlier use; kept as-is.
            mu_mle = np.exp(Xb @ mle)
            grad = np.zeros(d)
            for j in range(batch_size):
                xj = Xb[j].reshape(d, 1)
                grad += (mu_mle[j] * (xj @ xj.T) @ (theta - mle)).reshape(-1) / batch_size

        delta = gamma * (pre_mat @ grad)
        path[t + 1] = theta + delta

    return path


# =========================
# 5. Metrics
# =========================

def post_burnin_samples(path: np.ndarray, burnin_frac: float = 0.1, thin: int = 1) -> np.ndarray:
    T = path.shape[0]
    start = int(burnin_frac * T)
    return path[start::thin].copy()


def param_error(samples: np.ndarray, theta_target: np.ndarray, eps: float = 1e-12) -> float:
    mu = samples.mean(axis=0)
    return float(np.linalg.norm(mu - theta_target) / (np.linalg.norm(theta_target) + eps))


def cov_frob_error(samples: np.ndarray, cov_target: np.ndarray, eps: float = 1e-12) -> float:
    cov_hat = np.cov(samples.T, bias=True)
    return float(np.linalg.norm(cov_hat - cov_target, ord="fro") / (np.linalg.norm(cov_target, ord="fro") + eps))


def quantile_calib_rmse(samples: np.ndarray, mean: np.ndarray, cov: np.ndarray,
                        q_list=(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
                        jitter: float = 1e-10) -> float:
    d = cov.shape[0]
    cov_stable = cov + jitter * np.eye(d)
    L = np.linalg.cholesky(cov_stable)
    z = np.linalg.solve(L, (samples - mean).T).T  # whiten

    q = np.array(q_list)
    zq_target = norm.ppf(q)
    zq_emp = np.quantile(z, q, axis=0)

    diff = zq_emp - zq_target.reshape(-1, 1)
    rmse_dim = np.sqrt(np.mean(diff ** 2, axis=0))
    return float(np.mean(rmse_dim))


def poisson_test_nll(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ theta
    mu = np.exp(eta)
    return float(np.mean(mu - y * eta + gammaln(y + 1.0)))


def mean_pred_mse(theta: np.ndarray, X: np.ndarray, mu_true: np.ndarray) -> float:
    mu_hat = np.exp(X @ theta)
    return float(np.mean((mu_hat - mu_true) ** 2))


def metrics_from_path(
    path: np.ndarray,
    mle: np.ndarray,
    cov_target: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mu_true_test: np.ndarray,
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> Optional[Dict[str, float]]:
    samples = post_burnin_samples(path, burnin_frac, thin)
    if samples.shape[0] < 5 or (not is_finite(samples)):
        return None

    theta_bar = samples.mean(axis=0)
    return {
        "param_error": param_error(samples, mle),
        "quantile_calib_error": quantile_calib_rmse(samples, mle, cov_target),
        "cov_frob_error": cov_frob_error(samples, cov_target),
        "test_nll": poisson_test_nll(theta_bar, X_test, y_test),
        "test_mean_pred_error": mean_pred_mse(theta_bar, X_test, mu_true_test),
    }


# =========================
# 6. Fixed test set (so CI reflects algorithm randomness)
# =========================

def generate_poisson_test_set(N_test: int, cov_x_diag: np.ndarray, true_theta: np.ndarray, bias: float):
    d = len(cov_x_diag)
    X = np.random.multivariate_normal(mean=np.zeros(d), cov=np.diag(cov_x_diag), size=N_test)
    eta = X @ true_theta + bias
    mu = np.exp(eta)
    y = np.random.poisson(lam=mu)
    return X, y, mu


# =========================
# 7. Repeated runs + CI table
# =========================

METHODS_ORDER = [
    "continuous-time",
    "discrete-quadratic+constant noise",
    "taylor (this paper)",
]

METRICS = [
    "param_error",
    "quantile_calib_error",
    "cov_frob_error",
    "test_nll",
    "test_mean_pred_error",
]


def run_with_cis(
    n_reps: int,
    seed0: int,
    batch_sizes: List[int],
    num_epochs: int,
    X: np.ndarray,
    y: np.ndarray,
    initial_state: np.ndarray,
    lr_list_ct: List[float],
    pre_mat_ct: np.ndarray,
    Lambda_improved_by_idx: Dict[int, np.ndarray],
    Lambda_taylor_by_idx: Dict[int, np.ndarray],
    mle: np.ndarray,
    cov_target: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mu_true_test: np.ndarray,
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    N = X.shape[0]
    records = []

    for rep in range(n_reps):
        set_seed(seed0 + rep)

        for i, B in enumerate(batch_sizes):
            n_steps = num_epochs * int(N / B)

            # ---- CT
            path_ct = sgd_poisson(
                n_steps=n_steps, batch_size=B,
                theta0=initial_state, lr=lr_list_ct[i],
                pre_mat=pre_mat_ct, X=X, y=y, mle=mle,
                fixed_lr=True, approximate_loss=False
            )
            m = metrics_from_path(path_ct, mle, cov_target, X_test, y_test, mu_true_test, burnin_frac, thin)
            if m is not None:
                records.append({"rep": rep, "method": "continuous-time", "batch_size": int(B), **m})

            # ---- Improved
            path_im = sgd_poisson(
                n_steps=n_steps, batch_size=B,
                theta0=initial_state, lr=1.0,
                pre_mat=Lambda_improved_by_idx[i], X=X, y=y, mle=mle,
                fixed_lr=True, approximate_loss=False
            )
            m = metrics_from_path(path_im, mle, cov_target, X_test, y_test, mu_true_test, burnin_frac, thin)
            if m is not None:
                records.append({"rep": rep, "method": "discrete-quadratic+constant noise", "batch_size": int(B), **m})

            # ---- Taylor (this paper)
            path_ta = sgd_poisson(
                n_steps=n_steps, batch_size=B,
                theta0=initial_state, lr=1.0,
                pre_mat=Lambda_taylor_by_idx[i], X=X, y=y, mle=mle,
                fixed_lr=True, approximate_loss=False
            )
            m = metrics_from_path(path_ta, mle, cov_target, X_test, y_test, mu_true_test, burnin_frac, thin)
            if m is not None:
                records.append({"rep": rep, "method": "taylor (this paper)", "batch_size": int(B), **m})

    df_raw = pd.DataFrame(records)

    rows = []
    for (method, B), sub in df_raw.groupby(["method", "batch_size"]):
        row = {"method": method, "batch_size": int(B)}
        for m in METRICS:
            mean, lo, hi, n = summarize_ci(sub[m].values, alpha=0.05)
            row[f"{m}_mean"] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
            row[f"{m}_n"] = n
        rows.append(row)

    df_ci = (
        pd.DataFrame(rows)
        .sort_values(["batch_size", "method"])
        .reset_index(drop=True)
    )
    return df_raw, df_ci


def make_table_with_cis(df_ci: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = []
    for _, r in df_ci.sort_values(["batch_size", "method"]).iterrows():
        row = {"method": r["method"], "batch_size": int(r["batch_size"])}
        for m in METRICS:
            row[m] = format_ci(r[f"{m}_mean"], r[f"{m}_lo"], r[f"{m}_hi"], digits)
        out.append(row)
    return pd.DataFrame(out)


# =========================
# 8. Plotting
# =========================

def plot_metrics_with_cis(df_ci: pd.DataFrame, batch_sizes: List[int], savepath: Optional[str] = None):
    metrics_info = [
        ("param_error", "Relative parameter error"),
        ("quantile_calib_error", "Quantile-calibration RMSE"),
        ("cov_frob_error", "Relative covariance error (Frobenius)"),
        ("test_nll", "Test NLL"),
        ("test_mean_pred_error", "Test mean predictive error (MSE)"),
    ]

    plt.figure(figsize=(26, 4.5))
    plt.rcParams.update({"font.size": 13})

    for k, (mname, ylabel) in enumerate(metrics_info, start=1):
        plt.subplot(1, 5, k)
        for method in METHODS_ORDER:
            sub = df_ci[df_ci["method"] == method].sort_values("batch_size")
            x = sub["batch_size"].values.astype(float)
            y = sub[f"{mname}_mean"].values.astype(float)
            lo = sub[f"{mname}_lo"].values.astype(float)
            hi = sub[f"{mname}_hi"].values.astype(float)
            yerr = np.vstack([y - lo, hi - y])
            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=method)

        plt.xscale("log")
        plt.xlabel("batch size $B$")
        plt.ylabel(ylabel)
        plt.title(mname)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


def plot_cov_error_with_cis(df_ci: pd.DataFrame, savepath: Optional[str] = None):
    mname = "cov_frob_error"

    plt.figure(figsize=(6.5, 4.8))
    plt.rcParams.update({"font.size": 13})

    for method in METHODS_ORDER:
        sub = df_ci[df_ci["method"] == method].sort_values("batch_size")
        x = sub["batch_size"].values.astype(float)
        y = sub[f"{mname}_mean"].values.astype(float)
        lo = sub[f"{mname}_lo"].values.astype(float)
        hi = sub[f"{mname}_hi"].values.astype(float)
        yerr = np.vstack([y - lo, hi - y])
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=method)

    plt.xscale("log")
    plt.xlabel("batch size $B$")
    plt.ylabel("relative Frobenius error")
    plt.title("Covariance error (95% CIs)")
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


# =========================
# 9. Main (paper run)
# =========================

def main():
    # ---- Config
    cfg = SimPoissonConfig(N=2000, d_num=40, theta_sigma=0.1, bias=0.5)
    num_epochs = 500
    batch_sizes = [16, int(0.1 * cfg.N)]
    n_reps = 30
    seed0 = 123
    burnin_frac = 0.1
    thin = 1

    # ---- Generate data
    set_seed(0)
    X, y, true_theta, bias, cov_x_diag = generate_simulated_poisson(cfg)
    N, d = X.shape

    # ---- Fit MLE + sandwich target
    mle = fit_poisson_mle_statsmodels(X, y)
    J, V = compute_J_V_at_mle(X, y, mle)
    _, scaled_sandwich = sandwich_covariance_scaled(J, V, N, eps=0.0)

    # ---- Fixed test set (same DGP)
    X_test, y_test, mu_true_test = generate_poisson_test_set(
        N_test=2000, cov_x_diag=cov_x_diag, true_theta=true_theta, bias=bias
    )

    # ---- Preconditioners
    Jinv = np.linalg.inv(J)
    pre_mat_ct = precond_CT(Jinv)

    Lambda_improved_by_idx = {}
    for i, B in enumerate(batch_sizes):
        Lambda_improved_by_idx[i] = precond_improved_quadratic_constant_noise(J, V, Jinv, N)

    # Taylor quantities shared across B (C1,C2,P), then Lambda depends on B
    C1, C2, P = compute_taylor_quantities(X, y, mle, scaled_sandwich)
    Lambda_taylor_by_idx = {}
    for i, B in enumerate(batch_sizes):
        Lambda_taylor_by_idx[i] = precond_taylor(C1, C2, P, J, scaled_sandwich, batch_size=B)

    # ---- Learning rates for CT baseline (your rule 2B/N)
    lr_list_ct = [2.0 * B / N for B in batch_sizes]

    # ---- Initial state
    initial_state = mle.copy()

    # ---- Run + summarize
    df_raw, df_ci = run_with_cis(
        n_reps=n_reps,
        seed0=seed0,
        batch_sizes=batch_sizes,
        num_epochs=num_epochs,
        X=X, y=y,
        initial_state=initial_state,
        lr_list_ct=lr_list_ct,
        pre_mat_ct=pre_mat_ct,
        Lambda_improved_by_idx=Lambda_improved_by_idx,
        Lambda_taylor_by_idx=Lambda_taylor_by_idx,
        mle=mle,
        cov_target=scaled_sandwich,
        X_test=X_test, y_test=y_test, mu_true_test=mu_true_test,
        burnin_frac=burnin_frac,
        thin=thin,
    )

    print("\n=== Mean + 95% CI table ===")
    table_df = make_table_with_cis(df_ci, digits=3)
    print(table_df.to_string(index=False))

    # ---- Plots
    plot_metrics_with_cis(df_ci, batch_sizes, savepath="poisson_compare_five_metrics_CIs.pdf")
    plot_cov_error_with_cis(df_ci, savepath="poisson_cov_error_CIs.pdf")


if __name__ == "__main__":
    main()
