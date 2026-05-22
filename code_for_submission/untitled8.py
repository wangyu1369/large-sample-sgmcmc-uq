import time
import numpy as np
import pandas as pd

from utils import (
    linear_normal,
    nonlinear_exp,
    nonlinear_quadratic,
    nonlinear_dependent_noise,
)

# =========================
# Timing helpers
# =========================
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def inv_via_solve(M):
    """Compute M^{-1} via solve(M, I) for better numerical stability."""
    I = np.eye(M.shape[0])
    return np.linalg.solve(M, I)

# =========================
# SGD (your code, slightly cleaned)
# =========================
def sgd(n, batch_size, theta_0, lr_0, pre_matirx,
        fixed_lr=True,
        N=None, precondition=False, w1=1, w2=1,
        fixed_point=False, regulizer=False,
        x_raw=None, y_raw=None, theta_hat=None, dimension=None, model=None):
    """
    Runs SGD with (optional) preconditioning.
    Returns path array of shape (n+1, dimension).
    """
    if N is None:
        N = x_raw.shape[0]

    ret = np.zeros((n + 1, dimension))
    ret[0, :] = theta_0

    precondition_matirx = pre_matirx if precondition else np.eye(dimension)

    for i in range(n):
        gamma = w1 * lr_0 if fixed_lr else w1 * lr_0 / (i + 1)

        index = np.random.randint(0, N, batch_size)
        x_subsample = x_raw[index, :]
        y_subsample = y_raw[index]

        delta_theta = np.zeros((dimension, 1))
        pre_theta = ret[i, :].reshape(dimension, 1)

        for j in range(batch_size):
            x_j = x_subsample[j, :].reshape(dimension, 1)
            if fixed_point:
                delta_theta += (x_j @ x_j.T) @ (pre_theta - theta_hat.reshape(dimension, 1))
            else:
                delta_theta += (x_j @ x_j.T) @ pre_theta - x_j * y_subsample[j]

        delta_theta *= (gamma / batch_size)
        delta_theta = precondition_matirx @ delta_theta

        if not regulizer:
            ret[i + 1, :] = (pre_theta - delta_theta).ravel()
        else:
            ret[i + 1, :] = ((1 - model.penalty_para * gamma / N) * pre_theta - delta_theta).ravel()

    return ret

# =========================
# Preconditioner builders
# =========================
def compute_common_ingredients(x_raw, y_raw):
    """
    Computes MLE, V, sandwich, C1, C2, and other pieces needed for preconditioners.
    Returns a dict of ingredients.
    """
    N, d = x_raw.shape

    XtX = x_raw.T @ x_raw
    Xty = x_raw.T @ y_raw

    mle = np.linalg.solve(XtX, Xty)  # theta_hat

    predicted_y = x_raw @ mle
    errors = y_raw - predicted_y
    variance_noise = np.var(errors)

    A = XtX / N

    # V
    V = np.zeros((d, d))
    for i in range(N):
        xi = x_raw[i].reshape(d, 1)
        V += (errors[i] ** 2) * (xi @ xi.T)
    V /= N

    # Gamma / J
    XtX_inv = inv_via_solve(XtX)
    Gamma = N * XtX_inv
    J_inverse = Gamma
    J = inv_via_solve(J_inverse)

    sandwich = J_inverse @ V @ J_inverse
    scaled_sandwich = sandwich / N

    # E, C1, C2
    E_vector = (errors.T @ x_raw) / N
    E = E_vector.reshape(d, 1) @ E_vector.reshape(1, d)

    def calculate_c1(cov):
        res = np.zeros((d, d))
        for i in range(N):
            xi = x_raw[i].reshape(d, 1)
            G = xi @ xi.T
            res += G @ cov @ G
        res /= N
        res -= A @ cov @ A
        return res

    C1 = calculate_c1(cov=scaled_sandwich)

    C2 = np.zeros((d, d))
    for i in range(N):
        xi = x_raw[i].reshape(d, 1)
        C2 += (errors[i] ** 2) * (xi @ xi.T)
    C2 /= N
    C2 -= E

    return {
        "N": N,
        "d": d,
        "XtX": XtX,
        "mle": mle,
        "errors": errors,
        "variance_noise": variance_noise,
        "A": A,
        "V": V,
        "Gamma": Gamma,
        "J_inverse": J_inverse,
        "scaled_sandwich": scaled_sandwich,
        "C1": C1,
        "C2": C2,
    }

def build_Gamma(ing):
    return ing["Gamma"]

def build_Lambda_ls(ing, batch_size):
    """
    discrete-time quadratic loss + well-specified approximation
    """
    N = ing["N"]
    d = ing["d"]
    A = ing["A"]
    V = ing["V"]
    J_inverse = ing["J_inverse"]
    target_cov = ing["scaled_sandwich"]
    variance_noise = ing["variance_noise"]

    C = (A @ target_cov @ A + np.trace(A @ target_cov) * A) + (1 - d / N) * variance_noise * A
    C /= batch_size

    M = C + (1 / N) * V
    Minv = inv_via_solve(M)

    Lambda = (1 / N) * (V @ J_inverse + J_inverse @ V) @ Minv
    return Lambda

def build_Lambda_exact(ing, batch_size):
    """
    your exact guidance using C1 + C2
    """
    N = ing["N"]
    V = ing["V"]
    J_inverse = ing["J_inverse"]
    C1 = ing["C1"]
    C2 = ing["C2"]

    C = (C1 + C2) / batch_size
    M = C + (1 / N) * V
    Minv = inv_via_solve(M)

    Lambda = (1 / N) * (V @ J_inverse + J_inverse @ V) @ Minv
    return Lambda

# =========================
# Main benchmark
# =========================
def main():
    np.random.seed(0)

    dimension_list = [5, 10, 20, 50]
    sample_size = 1000
    batch_size = 64
    num_epochs = 500
    n_reps = 5  # change as you like

    # Preconditioners to compare
    METHODS = ["Gamma", "Lambda_ls", "Lambda_exact"]

    rows = []

    for d in dimension_list:
        for rep in range(n_reps):

            # ----- data generation (choose your model) -----
            model = nonlinear_dependent_noise(
                N=sample_size, rho=0, p=d, sigma=0.1, penalty_para=0,
                theta_loc=0, theta_sigma=1, bias=1
            )
            x_raw, y_raw = model.generate_data()
            y_raw = y_raw.reshape(y_raw.shape[1], )

            # ----- common ingredients (not timed as "precondition matrix") -----
            ing = compute_common_ingredients(x_raw, y_raw)

            N = ing["N"]
            mle = ing["mle"]

            # number of iterations for 50 epochs at batch size 64
            n_iters = num_epochs * (N // batch_size)

            # ----- build + time each preconditioner, then time SGD path -----
            for method in METHODS:

                if method == "Gamma":
                    preM, t_pre = timed(build_Gamma, ing)
                    lr = 2.0 * batch_size / N   # your CT rule
                elif method == "Lambda_ls":
                    preM, t_pre = timed(build_Lambda_ls, ing, batch_size)
                    lr = 1.0
                elif method == "Lambda_exact":
                    preM, t_pre = timed(build_Lambda_exact, ing, batch_size)
                    lr = 1.0
                else:
                    raise ValueError(f"Unknown method: {method}")

                # time "MCMC" (SGD path)
                _, t_mcmc = timed(
                    sgd,
                    n_iters, batch_size, mle, lr, preM,
                    True, N, True, 1, 1, False, False,
                    x_raw, y_raw, mle, ing["d"], model
                )

                rows.append({
                    "dim": d,
                    "rep": rep,
                    "method": method,
                    "precond_time_sec": t_pre,
                    "mcmc_time_sec": t_mcmc,
                    "total_time_sec": t_pre + t_mcmc,
                    "N": N,
                    "batch_size": batch_size,
                    "epochs": num_epochs,
                    "iters": n_iters,
                })

            print(f"done: dim={d}, rep={rep}")

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["dim", "method"])[["precond_time_sec", "mcmc_time_sec", "total_time_sec"]]
          .agg(["mean", "std"])
          .reset_index()
    )

    print("\n=== RAW RESULTS ===")
    print(df)

    print("\n=== SUMMARY (mean ± std over reps) ===")
    print(summary)

    # Save outputs
    df.to_csv("timing_raw.csv", index=False)
    summary.to_csv("timing_summary.csv", index=False)
    print("\nSaved: timing_raw.csv, timing_summary.csv")

if __name__ == "__main__":
    main()
