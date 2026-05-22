import time
import numpy as np
import pandas as pd
import statsmodels.api as sm

# =========================
# Timing helpers
# =========================
def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def inv_via_solve(M, eps=0.0):
    """Compute (M + eps I)^{-1} via solve for stability."""
    d = M.shape[0]
    I = np.eye(d)
    if eps > 0:
        return np.linalg.solve(M + eps * I, I)
    return np.linalg.solve(M, I)

# =========================
# Data model (your Poisson regression)
# =========================
class poisson_regression:
    def __init__(self, x, theta_loc=0, theta_sigma=0.1, bias=0.5):
        self.x = x
        self.N = x.shape[0]
        self.p = x.shape[1]
        self.bias = bias
        self.true_theta = np.random.normal(loc=theta_loc, scale=theta_sigma, size=self.p)

    def generate_data(self):
        x = self.x
        y = np.random.poisson(lam=np.exp(x.dot(self.true_theta) + self.bias))
        return x, y

# =========================
# SGD (Poisson version; close to yours)
# =========================
def sgd_poisson(n, batch_size, theta_0, lr_0, pre_matrix,
               x_raw, y_raw, mle,
               fixed_lr=True, precondition=True,
               approximate_loss=False, fixed_point=False, w1=1):
    """
    Returns path array shape (n+1, d).
    Uses your "approximate_loss" switch; default False (true Poisson gradient).
    """
    N, d = x_raw.shape
    ret = np.zeros((n + 1, d))
    ret[0, :] = theta_0

    M = pre_matrix if precondition else np.eye(d)

    for t in range(n):
        gamma = w1 * lr_0 if fixed_lr else w1 * lr_0 / (t + 1)

        idx = np.random.randint(0, N, batch_size)
        x_sub = x_raw[idx, :]
        y_sub = y_raw[idx]

        pre_theta = ret[t, :].reshape(d, 1)
        delta = np.zeros((d, 1))

        for j in range(batch_size):
            xj = x_sub[j, :].reshape(d, 1)

            if not approximate_loss:
                if not fixed_point:
                    # stochastic gradient of log-likelihood (scaled like your code)
                    delta += (y_sub[j] - np.exp((xj.T @ pre_theta).item())) * xj / batch_size
                else:
                    delta += (np.exp((xj.T @ mle.reshape(d,1)).item()) - np.exp((xj.T @ pre_theta).item())) * xj / batch_size
            else:
                # your quadratic surrogate around MLE
                delta += np.exp((xj.T @ mle.reshape(d,1)).item()) * (xj @ xj.T) @ (pre_theta - mle.reshape(d,1))

        delta *= gamma
        delta = M @ delta

        ret[t + 1, :] = (pre_theta + delta).ravel()

    return ret

# =========================
# Ingredient computations (J, V, sandwich, Taylor terms)
# =========================
def compute_mle_statsmodels(x_raw, y_raw):
    # Use statsmodels GLM Poisson, as in your code.
    poisson_model = sm.GLM(y_raw, x_raw, family=sm.families.Poisson())
    res = poisson_model.fit()
    return res.params

def compute_core_matrices(x_raw, y_raw, mle):
    """
    Computes J, V, J^{-1}, scaled sandwich, derivative-related terms for Taylor preconditioner.
    Matches your code logic.
    """
    N, d = x_raw.shape

    J = np.zeros((d, d))
    V = np.zeros((d, d))

    mu = np.exp(x_raw @ mle)  # vector length N

    for i in range(N):
        xi = x_raw[i].reshape(d, 1)
        J += mu[i] * (xi @ xi.T)
        V += (y_raw[i] - mu[i])**2 * (xi @ xi.T)

    J /= N
    V /= N

    # preconditioner (Fisher inverse)
    J_inv = inv_via_solve(J, eps=1e-12)

    sandwich = J_inv @ V @ J_inv
    scaled_sandwich = sandwich / N

    return J, V, J_inv, scaled_sandwich

def compute_taylor_terms(x_raw, y_raw, mle, scaled_sandwich):
    """
    Computes C1, C2, Derivative_vector, P needed for Lambda_taylor
    following your first-order Taylor section.
    """
    N, d = x_raw.shape
    mu = np.exp(x_raw @ mle)
    errors = y_raw - mu

    # E
    E_vec = (errors.T @ x_raw) / N
    E = E_vec.reshape(d, 1) @ E_vec.reshape(1, d)

    # derivatives at MLE: exp(x_i^T mle)
    deriv = mu  # length N

    Derivative_vector = np.zeros((d, d))
    for i in range(N):
        xi = x_raw[i].reshape(d, 1)
        Derivative_vector += deriv[i] * (xi @ xi.T)
    Derivative_vector /= N

    # C1
    def calculate_c1(cov):
        res = np.zeros((d, d))
        for i in range(N):
            xi = x_raw[i].reshape(d, 1)
            G = xi @ xi.T
            res += (deriv[i]**2) * (G @ cov @ G)
        res /= N
        res -= Derivative_vector @ cov @ Derivative_vector
        return res

    C1 = calculate_c1(scaled_sandwich)

    # C2
    C2 = np.zeros((d, d))
    for i in range(N):
        xi = x_raw[i].reshape(d, 1)
        C2 += (errors[i] ** 2) * (xi @ xi.T)
    C2 /= N
    C2 -= E

    # P
    P = scaled_sandwich @ Derivative_vector

    return C1, C2, Derivative_vector, P

# =========================
# Preconditioners (timed)
# =========================
def build_precond_Jinv(J):
    return inv_via_solve(J, eps=1e-12)

def build_precond_Lambda_ls(J, V, J_inv, eps=1e-10):
    """
    Your "Improved preconditioned matrix" block uses C=J (not divided by B in your final),
    and Lambda=(1/N)(VJ^{-1}+J^{-1}V)(J+(1/N)V)^{-1}.
    """
    d = J.shape[0]
    # Note: J and V here are already averaged by N in compute_core_matrices
    # The formula still uses (1/N)*V term; we need N externally to be exact.
    # We'll pass N via closure when calling; see build_lambda_ls below.
    raise RuntimeError("Use build_precond_Lambda_ls_with_N(J, V, J_inv, N)")

def build_precond_Lambda_ls_with_N(J, V, J_inv, N, eps=1e-10):
    d = J.shape[0]
    M = J + (1 / N) * V + eps * np.eye(d)
    Minv = inv_via_solve(M)
    Lambda = (1 / N) * (V @ J_inv + J_inv @ V) @ Minv
    return Lambda

def build_precond_Lambda_taylor(C1, C2, Derivative_vector, P, scaled_sandwich, batch_size, eps=1e-10):
    """
    Matches your:
      C = (C1+C2)/B
      Lambda = (P+P^T) @ inv( C + Derivative_vector @ S @ Derivative_vector )
    """
    d = C1.shape[0]
    C = (C1 + C2) / batch_size
    M = C + Derivative_vector @ scaled_sandwich @ Derivative_vector + eps * np.eye(d)
    Minv = inv_via_solve(M)
    Lambda = (P + P.T) @ Minv
    return Lambda

# =========================
# Benchmark
# =========================
def main():
    np.random.seed(0)

    # --- benchmark settings
    dimension_list = [5, 10, 20, 50]
    sample_size = 2000
    batch_size = 64
    num_epochs = 500
    n_reps = 3  # increase to 5-10 for smoother timing

    # Data generation: diagonal covariance with varying variances, like your code
    def make_cov_diag(d, N, variance_list=(1, 2, 3, 4, 5)):
        cov_x = np.ones(d)
        for i in range(d):
            cov_x[i] = 1 / (np.random.choice(variance_list) * N)
        return np.diag(cov_x)

    METHODS = ["J_inv", "Lambda_ls", "Lambda_taylor"]

    rows = []

    for d in dimension_list:
        for rep in range(n_reps):
            # ----- generate x and y -----
            cov = make_cov_diag(d, sample_size)
            x_raw = np.random.multivariate_normal(mean=np.zeros(d), cov=cov, size=sample_size)

            model = poisson_regression(x=x_raw, theta_loc=0, theta_sigma=0.1, bias=0.5)
            x_raw, y_raw = model.generate_data()
            N, d = x_raw.shape

            # ----- MLE (not counted as "precondition matrix time" by default)
            mle = compute_mle_statsmodels(x_raw, y_raw)

            # ----- core matrices (J,V,etc.) (also not counted as "precondition time" by default)
            J, V, J_inv_core, scaled_sandwich = compute_core_matrices(x_raw, y_raw, mle)

            # ----- Taylor terms (needed for Lambda_taylor)
            C1, C2, Derivative_vector, P = compute_taylor_terms(x_raw, y_raw, mle, scaled_sandwich)

            # iterations for 50 epochs
            n_iters = num_epochs * (N // batch_size)

            # ----- timed preconditioners + timed SGD paths
            for method in METHODS:
                if method == "J_inv":
                    preM, t_pre = timed(build_precond_Jinv, J)
                    lr = 2.0 * batch_size / N  # your CT choice
                elif method == "Lambda_ls":
                    preM, t_pre = timed(build_precond_Lambda_ls_with_N, J, V, J_inv_core, N)
                    lr = 1.0
                elif method == "Lambda_taylor":
                    preM, t_pre = timed(
                        build_precond_Lambda_taylor,
                        C1, C2, Derivative_vector, P, scaled_sandwich, batch_size
                    )
                    lr = 1.0
                else:
                    raise ValueError(method)

                # time the MCMC/SGD run
                _, t_mcmc = timed(
                    sgd_poisson,
                    n_iters, batch_size, mle, lr, preM,
                    x_raw, y_raw, mle,
                    True, True, False, False, 1
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

    print("\n=== SUMMARY (mean ± std) ===")
    print(summary)

    df.to_csv("timing_poisson_raw.csv", index=False)
    summary.to_csv("timing_poisson_summary.csv", index=False)
    print("\nSaved: timing_poisson_raw.csv, timing_poisson_summary.csv")

if __name__ == "__main__":
    main()
