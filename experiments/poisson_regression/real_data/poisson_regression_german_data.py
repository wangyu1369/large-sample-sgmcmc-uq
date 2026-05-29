import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional, List

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# 0. Utilities
# =========================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def is_finite(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def summarize_ci(values, alpha: float = 0.05):
    """
    Percentile CI.
    alpha=0.05 gives a 95% CI.
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
# 1. German Credit data loader
# =========================

def load_german_credit(
    path: str,
    target_col: str = "Creditability",
    add_intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load German Credit data from german.csv or german.cvs.

    This function:
      1. detects the delimiter automatically,
      2. uses target_col as the response,
      3. one-hot encodes categorical covariates,
      4. standardizes features,
      5. optionally adds an intercept column.

    The response is converted to nonnegative numeric values for Poisson GLM.
    If labels are {1, 2}, they are mapped to {1, 0}, treating
    Creditability == 1 as the positive class.
    """
    df = pd.read_csv(path, sep=None, engine="python")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns are: {list(df.columns)}"
        )

    y_raw = df[target_col].to_numpy()
    X_df = df.drop(columns=[target_col])

    y_unique = np.sort(pd.unique(y_raw))

    if set(y_unique) == {1, 2}:
        y = (y_raw == 1).astype(float)
    else:
        y = pd.to_numeric(pd.Series(y_raw), errors="raise").to_numpy(dtype=float)

        if np.min(y) < 0:
            raise ValueError("Poisson GLM requires nonnegative response values.")

    X_df = pd.get_dummies(X_df, drop_first=True)

    X = X_df.to_numpy(dtype=float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if add_intercept:
        X = np.column_stack([np.ones(X.shape[0]), X])

    return X, y.astype(float)


# =========================
# 2. MLE + sandwich target
# =========================

def fit_poisson_mle_statsmodels(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Poisson GLM MLE using statsmodels.
    """
    model = sm.GLM(y, X, family=sm.families.Poisson())
    res = model.fit(maxiter=200)
    return res.params


def compute_J_V_at_mle(
    X: np.ndarray,
    y: np.ndarray,
    mle: np.ndarray,
    clip: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    J = E[ exp(x^T mle) x x^T ]
    V = E[ (y - exp(x^T mle))^2 x x^T ]

    Both are sample averages.
    """
    N, _ = X.shape

    eta = np.clip(X @ mle, -clip, clip)
    mu = np.exp(eta)

    J = (X.T * mu) @ X / N

    r2 = (y - mu) ** 2
    V = (X.T * r2) @ X / N

    J = 0.5 * (J + J.T)
    V = 0.5 * (V + V.T)

    return J, V


def sandwich_covariance_scaled(
    J: np.ndarray,
    V: np.ndarray,
    N: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      sandwich = J^{-1} V J^{-1}
      scaled_sandwich = sandwich / N
    """
    d = J.shape[0]
    J_reg = J + eps * np.eye(d)

    Jinv = np.linalg.inv(J_reg)

    sandwich = Jinv @ V @ Jinv
    sandwich = 0.5 * (sandwich + sandwich.T)

    scaled_sandwich = sandwich / N
    scaled_sandwich = 0.5 * (scaled_sandwich + scaled_sandwich.T)

    return sandwich, scaled_sandwich


# =========================
# 3. Preconditioners
# =========================

def precond_CT(Jinv: np.ndarray) -> np.ndarray:
    """
    CT preconditioner.
    """
    return Jinv


def precond_improved_quadratic_constant_noise(
    J: np.ndarray,
    V: np.ndarray,
    Jinv: np.ndarray,
    N: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    DQ+const preconditioner.

        Lambda = (1/N) (V J^{-1} + J^{-1} V) (C + V/N)^{-1},

    with C = J.
    """
    d = J.shape[0]

    C = J
    M = C + (1.0 / N) * V + eps * np.eye(d)
    M = 0.5 * (M + M.T)

    Lambda = (1.0 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(M)

    return Lambda


def compute_taylor_quantities(
    X: np.ndarray,
    y: np.ndarray,
    mle: np.ndarray,
    scaled_sandwich: np.ndarray,
    clip: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes C1, C2, P for the DQ+Exact preconditioner.
    """
    N, d = X.shape

    eta = np.clip(X @ mle, -clip, clip)
    mu = np.exp(eta)

    J = (X.T * mu) @ X / N
    J = 0.5 * (J + J.T)

    errors = y - mu

    Ex = (errors @ X) / N
    E = Ex.reshape(d, 1) @ Ex.reshape(1, d)

    C2 = (X.T * (errors ** 2)) @ X / N - E
    C2 = 0.5 * (C2 + C2.T)

    C1 = np.zeros((d, d))

    for i in range(N):
        xi = X[i].reshape(d, 1)
        xxT = xi @ xi.T
        C1 += (mu[i] ** 2) * (xxT @ scaled_sandwich @ xxT)

    C1 /= N
    C1 -= J @ scaled_sandwich @ J
    C1 = 0.5 * (C1 + C1.T)

    P = scaled_sandwich @ J

    return C1, C2, P


def precond_taylor(
    C1: np.ndarray,
    C2: np.ndarray,
    P: np.ndarray,
    J: np.ndarray,
    scaled_sandwich: np.ndarray,
    batch_size: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    DQ+Exact preconditioner.

        C = (C1 + C2) / B
        Lambda = (P + P^T) (C + J cov J)^{-1}
    """
    d = J.shape[0]

    C = (C1 + C2) / float(batch_size)
    denom = C + J @ scaled_sandwich @ J
    denom = 0.5 * (denom + denom.T) + eps * np.eye(d)

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
    clip: float = 30.0,
) -> np.ndarray:
    """
    Preconditioned stochastic-gradient updates for Poisson regression.

    If approximate_loss=False:
        uses Poisson minibatch score.

    If approximate_loss=True:
        uses quadratic approximation around mle.
    """
    N, d = X.shape

    path = np.zeros((n_steps + 1, d))
    path[0] = theta0.copy()

    for t in range(n_steps):
        gamma = lr if fixed_lr else lr / (t + 1)

        idx = np.random.randint(0, N, size=batch_size)
        Xb = X[idx]
        yb = y[idx]

        theta = path[t]

        if not approximate_loss:
            eta = np.clip(Xb @ theta, -clip, clip)
            mu = np.exp(eta)

            # Score convention: grad = (y - mu) x.
            grad = ((yb - mu)[:, None] * Xb).mean(axis=0)
        else:
            mu_mle = np.exp(np.clip(Xb @ mle, -clip, clip))
            grad = np.zeros(d)

            for j in range(batch_size):
                xj = Xb[j].reshape(d, 1)
                grad += (
                    mu_mle[j]
                    * (xj @ xj.T)
                    @ (theta - mle)
                ).reshape(-1) / batch_size

        delta = gamma * (pre_mat @ grad)
        path[t + 1] = theta + delta

        if not np.isfinite(path[t + 1]).all():
            path = path[: t + 2]
            break

    return path


# =========================
# 5. Metrics
# =========================

def post_burnin_samples(
    path: np.ndarray,
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> np.ndarray:
    T = path.shape[0]
    start = int(burnin_frac * T)
    return path[start::thin].copy()


def cov_frob_error(
    samples: np.ndarray,
    cov_target: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Relative Frobenius covariance error.
    """
    cov_hat = np.cov(samples.T, bias=True)
    cov_hat = 0.5 * (cov_hat + cov_hat.T)

    return float(
        np.linalg.norm(cov_hat - cov_target, ord="fro")
        / (np.linalg.norm(cov_target, ord="fro") + eps)
    )


def metrics_from_path(
    path: np.ndarray,
    mle: np.ndarray,
    cov_target: np.ndarray,
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> Optional[Dict[str, float]]:
    """
    Compute only covariance error.
    """
    samples = post_burnin_samples(path, burnin_frac, thin)

    if samples.shape[0] < 5 or not is_finite(samples):
        return None

    return {
        "cov_frob_error": cov_frob_error(samples, cov_target),
    }


# =========================
# 6. Repeated runs + CI table
# =========================

METHODS_ORDER = [
    "CT",
    "DQ+const",
    "DQ+Exact",
]

METRICS = [
    "cov_frob_error",
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
    burnin_frac: float = 0.1,
    thin: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run repeated experiments and summarize covariance error mean + 95% CI.
    """
    N = X.shape[0]
    records = []

    for rep in range(n_reps):
        set_seed(seed0 + rep)

        print(f"\nReplicate {rep + 1}/{n_reps}")

        for i, B in enumerate(batch_sizes):
            n_steps = num_epochs * int(N / B)

            # ---- CT
            path_ct = sgd_poisson(
                n_steps=n_steps,
                batch_size=B,
                theta0=initial_state,
                lr=lr_list_ct[i],
                pre_mat=pre_mat_ct,
                X=X,
                y=y,
                mle=mle,
                fixed_lr=True,
                approximate_loss=False,
            )

            m = metrics_from_path(
                path_ct,
                mle,
                cov_target,
                burnin_frac,
                thin,
            )

            if m is not None:
                records.append({
                    "rep": rep,
                    "method": "CT",
                    "batch_size": int(B),
                    **m,
                })

            # ---- DQ+const
            path_im = sgd_poisson(
                n_steps=n_steps,
                batch_size=B,
                theta0=initial_state,
                lr=1.0,
                pre_mat=Lambda_improved_by_idx[i],
                X=X,
                y=y,
                mle=mle,
                fixed_lr=True,
                approximate_loss=False,
            )

            m = metrics_from_path(
                path_im,
                mle,
                cov_target,
                burnin_frac,
                thin,
            )

            if m is not None:
                records.append({
                    "rep": rep,
                    "method": "DQ+const",
                    "batch_size": int(B),
                    **m,
                })

            # ---- DQ+Exact
            path_ta = sgd_poisson(
                n_steps=n_steps,
                batch_size=B,
                theta0=initial_state,
                lr=1.0,
                pre_mat=Lambda_taylor_by_idx[i],
                X=X,
                y=y,
                mle=mle,
                fixed_lr=True,
                approximate_loss=False,
            )

            m = metrics_from_path(
                path_ta,
                mle,
                cov_target,
                burnin_frac,
                thin,
            )

            if m is not None:
                records.append({
                    "rep": rep,
                    "method": "DQ+Exact",
                    "batch_size": int(B),
                    **m,
                })

    df_raw = pd.DataFrame(records)

    rows = []
    for (method, B), sub in df_raw.groupby(["method", "batch_size"]):
        row = {
            "method": method,
            "batch_size": int(B),
        }

        mean, lo, hi, n = summarize_ci(sub["cov_frob_error"].values, alpha=0.05)

        row["cov_frob_error_mean"] = mean
        row["cov_frob_error_lo"] = lo
        row["cov_frob_error_hi"] = hi
        row["cov_frob_error_n"] = n

        rows.append(row)

    method_rank = {m: k for k, m in enumerate(METHODS_ORDER)}

    df_ci = pd.DataFrame(rows)
    df_ci["method_rank"] = df_ci["method"].map(method_rank)

    df_ci = (
        df_ci
        .sort_values(["batch_size", "method_rank"])
        .drop(columns=["method_rank"])
        .reset_index(drop=True)
    )

    return df_raw, df_ci


def make_table_with_cis(
    df_ci: pd.DataFrame,
    digits: int = 3,
) -> pd.DataFrame:
    """
    Public-facing covariance error table with mean + 95% CI.
    """
    method_rank = {m: k for k, m in enumerate(METHODS_ORDER)}

    df_show = df_ci.copy()
    df_show["method_rank"] = df_show["method"].map(method_rank)

    out = []
    for _, r in df_show.sort_values(["batch_size", "method_rank"]).iterrows():
        row = {
            "batch_size": int(r["batch_size"]),
            "method": r["method"],
            "covariance_error": format_ci(
                r["cov_frob_error_mean"],
                r["cov_frob_error_lo"],
                r["cov_frob_error_hi"],
                digits,
            ),
        }

        out.append(row)

    return pd.DataFrame(out)


def make_table_means_only(
    df_ci: pd.DataFrame,
    digits: int = 3,
) -> pd.DataFrame:
    """
    Mean-only covariance error table for main-text use.
    """
    method_rank = {m: k for k, m in enumerate(METHODS_ORDER)}

    df_show = df_ci.copy()
    df_show["method_rank"] = df_show["method"].map(method_rank)

    out = []
    for _, r in df_show.sort_values(["batch_size", "method_rank"]).iterrows():
        val = r["cov_frob_error_mean"]

        row = {
            "batch_size": int(r["batch_size"]),
            "method": r["method"],
            "covariance_error": f"{val:.{digits}f}" if np.isfinite(val) else "NA",
        }

        out.append(row)

    return pd.DataFrame(out)


# =========================
# 7. Plotting
# =========================

def plot_cov_error_with_cis(
    df_ci: pd.DataFrame,
    savepath: Optional[str] = None,
):
    """
    Plot covariance error only.
    """
    metric_name = "cov_frob_error"

    plt.figure(figsize=(6.5, 4.8))
    plt.rcParams.update({"font.size": 13})

    for method in METHODS_ORDER:
        sub = df_ci[df_ci["method"] == method].sort_values("batch_size")

        x = sub["batch_size"].values.astype(float)
        y = sub[f"{metric_name}_mean"].values.astype(float)
        lo = sub[f"{metric_name}_lo"].values.astype(float)
        hi = sub[f"{metric_name}_hi"].values.astype(float)

        yerr = np.vstack([y - lo, hi - y])

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xscale("log")
    plt.xlabel("Batch size $B$")
    plt.ylabel("Covariance error")
    plt.title("Covariance error")
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")

    plt.show()


# =========================
# 8. Main experiment
# =========================

if __name__ == "__main__":

    # ---- Config
    # Use "german.cvs" here if your file is actually named german.cvs.
    data_path = "german.csv"
    target_col = "Creditability"

    num_epochs = 500
    n_reps = 30
    seed0 = 123
    burnin_frac = 0.1
    thin = 1

    # ---- Load German Credit data
    X_all, y_all = load_german_credit(
        path=data_path,
        target_col=target_col,
        add_intercept=True,
    )

    X, X_test, y, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=seed0,
        stratify=y_all if len(np.unique(y_all)) > 1 else None,
    )

    N, d = X.shape

    print(f"Loaded German Credit data from: {data_path}")
    print(f"Train size: N={N}, dimension d={d}")
    print(f"Test size: {X_test.shape[0]}")
    print(f"Response values: {np.unique(y_all)}")

    # ---- Batch sizes
    batch_sizes = [
        16,
        int(0.1 * N),
    ]

    batch_sizes = [
        B for B in batch_sizes
        if B >= 2 and B < N
    ]

    print(f"Batch sizes: {batch_sizes}")

    # ---- Fit MLE + sandwich target
    mle = fit_poisson_mle_statsmodels(X, y)

    J, V = compute_J_V_at_mle(X, y, mle)

    _, scaled_sandwich = sandwich_covariance_scaled(
        J,
        V,
        N,
        eps=1e-8,
    )

    # ---- Preconditioners
    J_reg = J + 1e-8 * np.eye(d)
    Jinv = np.linalg.inv(J_reg)

    pre_mat_ct = precond_CT(Jinv)

    Lambda_improved_by_idx = {}
    for i, B in enumerate(batch_sizes):
        Lambda_improved_by_idx[i] = precond_improved_quadratic_constant_noise(
            J=J,
            V=V,
            Jinv=Jinv,
            N=N,
            eps=1e-8,
        )

    C1, C2, P = compute_taylor_quantities(
        X=X,
        y=y,
        mle=mle,
        scaled_sandwich=scaled_sandwich,
    )

    Lambda_taylor_by_idx = {}
    for i, B in enumerate(batch_sizes):
        Lambda_taylor_by_idx[i] = precond_taylor(
            C1=C1,
            C2=C2,
            P=P,
            J=J,
            scaled_sandwich=scaled_sandwich,
            batch_size=B,
            eps=1e-8,
        )

    # ---- Learning rates for CT baseline
    # Rule: lr = 2B/N.
    lr_list_ct = [
        2.0 * B / N
        for B in batch_sizes
    ]

    # ---- Initial state
    initial_state = mle.copy()

    # ---- Run repeated experiment
    df_raw, df_ci = run_with_cis(
        n_reps=n_reps,
        seed0=seed0,
        batch_sizes=batch_sizes,
        num_epochs=num_epochs,
        X=X,
        y=y,
        initial_state=initial_state,
        lr_list_ct=lr_list_ct,
        pre_mat_ct=pre_mat_ct,
        Lambda_improved_by_idx=Lambda_improved_by_idx,
        Lambda_taylor_by_idx=Lambda_taylor_by_idx,
        mle=mle,
        cov_target=scaled_sandwich,
        burnin_frac=burnin_frac,
        thin=thin,
    )

    # ---- Print covariance-error tables
    print("\n=== German Credit: Covariance error, mean + 95% CI ===")
    table_df = make_table_with_cis(df_ci, digits=3)
    print(table_df.to_string(index=False))

    print("\n=== German Credit: Covariance error, mean only ===")
    table_mean_df = make_table_means_only(df_ci, digits=3)
    print(table_mean_df.to_string(index=False))

    # ---- Save results
    df_raw.to_csv("german_credit_raw_results.csv", index=False)
    df_ci.to_csv("german_credit_covariance_error_summary_with_cis.csv", index=False)
    table_df.to_csv("german_credit_covariance_error_table_with_cis.csv", index=False)
    table_mean_df.to_csv("german_credit_covariance_error_table_means_only.csv", index=False)

    # ---- Plot
    plot_cov_error_with_cis(
        df_ci,
        savepath="german_credit_covariance_error_CIs.pdf",
    )

    print("\nSaved:")
    print("  german_credit_raw_results.csv")
    print("  german_credit_covariance_error_summary_with_cis.csv")
    print("  german_credit_covariance_error_table_with_cis.csv")
    print("  german_credit_covariance_error_table_means_only.csv")
    print("  german_credit_covariance_error_CIs.pdf")
