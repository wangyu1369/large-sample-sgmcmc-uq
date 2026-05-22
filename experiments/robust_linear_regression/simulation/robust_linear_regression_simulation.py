import autograd.numpy as np
from autograd import grad, hessian
from autograd.scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt


# ------------------------------
# Data generator (heteroskedastic + optional outliers)
# ------------------------------
class NonlinearDependentNoise:
    def __init__(self, N=200, p=5, rho=0.0, penalty_para=0.0,
                 theta_loc=0.0, theta_sigma=1.0, variance_list=(1, 2, 3), bias=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.p = int(p)
        self.N = int(N)
        self.rho = float(rho)
        self.penalty_para = float(penalty_para)
        self.bias = float(bias)

        self.true_theta = np.array([np.random.normal(theta_loc, theta_sigma) for _ in range(self.p)])

        cov_theta = np.zeros((self.p, self.p))
        for i in range(self.p):
            cov_theta[i, i] = np.random.choice(variance_list)
        for i in range(self.p):
            for j in range(self.p):
                if i != j:
                    cov_theta[i, j] = rho * np.sqrt(cov_theta[i, i]) * np.sqrt(cov_theta[j, j])

        # same as your code: cov_x depends on inv(cov_theta) and N
        self.cov_x = (np.linalg.inv(cov_theta) - penalty_para * np.eye(self.p)) / self.N

    def generate_data(self, outlier_frac=0.0, outlier_scale=20.0, outlier_shift=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)

        mean = np.zeros(self.p)
        X = np.random.multivariate_normal(mean, self.cov_x, self.N)

        # heteroskedastic variance: 1 + ||x||^2
        var_eps = np.array([1.0 + np.sum(X[i] ** 2) for i in range(self.N)])
        mean_y = X.dot(self.true_theta) + self.bias
        Y = np.random.normal(mean_y, np.sqrt(var_eps))

        # inject response outliers
        if outlier_frac > 0:
            n_out = int(outlier_frac * self.N)
            idx = np.random.choice(self.N, size=n_out, replace=False)
            Y[idx] = outlier_shift + np.random.normal(mean_y[idx], outlier_scale * np.sqrt(var_eps[idx]))

        return X, Y


# ============================================================
# Losses
# ============================================================

# --- beta-divergence loss uses a Gaussian "working likelihood" f(y|x,theta)=N(x^T theta, sigma^2) ---
def f_density(y, x, theta, sigma=1.0):
    mu = np.dot(x, theta)
    return norm.pdf(y, mu, sigma)

def beta_loss(theta, X, Y, beta=1.5, sigma=1.0, n_mc=200):
    """
    Empirical beta-divergence objective (up to constants).
    """
    N, _ = X.shape
    z_samples = np.random.normal(0, sigma * 5, size=n_mc)

    total = 0.0
    for i in range(N):
        fy = f_density(Y[i], X[i], theta, sigma)
        mu_i = np.dot(X[i], theta)
        fz = norm.pdf(z_samples, mu_i, sigma)
        integral_term = np.mean(fz ** beta)
        total += (- fy ** (beta - 1) / (beta - 1) + integral_term / beta)

    return total / N

def beta_loss_per_sample(theta, y, x, beta=1.5, sigma=1.0, n_mc=200):
    fy = f_density(y, x, theta, sigma)
    z_samples = np.random.normal(0, sigma * 5, size=n_mc)
    fz = f_density(z_samples, x, theta, sigma)
    integral_term = np.mean(fz ** beta)
    return - fy ** (beta - 1) / (beta - 1) + integral_term / beta


# --- NLL loss: Gaussian negative log-likelihood with fixed sigma ---
def nll_loss(theta, X, Y, sigma=1.0):
    r = (Y - X @ theta) / sigma
    # constants don't matter for gradients/Hessians, but keep for readability
    return np.mean(0.5 * r**2 + np.log(sigma) + 0.5 * np.log(2*np.pi))

def nll_loss_per_sample(theta, y, x, sigma=1.0):
    r = (y - np.dot(x, theta)) / sigma
    return 0.5 * r**2 + np.log(sigma) + 0.5 * np.log(2*np.pi)


# ============================================================
# Optimizer (for beta "MLE"/MAP)
# ============================================================
def rmsprop(loss_fn, theta_init, X, Y, lr=0.05, decay=0.9, eps=1e-8, n_iter=200, print_every=50, **loss_kwargs):
    theta = np.copy(theta_init)
    g_sq = np.zeros_like(theta)
    loss_grad = grad(lambda th, XX, YY: loss_fn(th, XX, YY, **loss_kwargs))

    for t in range(n_iter):
        g = loss_grad(theta, X, Y)
        g_sq = decay * g_sq + (1 - decay) * (g ** 2)
        theta -= lr * g / (np.sqrt(g_sq) + eps)
        if (print_every is not None) and (t % print_every == 0):
            L = loss_fn(theta, X, Y, **loss_kwargs)
            print(f"RMSProp iter {t:04d} | loss = {L:.6f}")
    return theta


# ============================================================
# J and V for beta and nll
# ============================================================
def empirical_fisher_from_per_sample_grad(theta_hat, X, Y, per_sample_loss_fn, per_sample_kwargs):
    """
    V = E[ g_i g_i^T ] where g_i = ∇_θ ℓ_i(θ)
    """
    N, D = X.shape
    g_fn = grad(lambda th, yi, xi: per_sample_loss_fn(th, yi, xi, **per_sample_kwargs))
    V = np.zeros((D, D))
    for i in range(N):
        g = g_fn(theta_hat, Y[i], X[i]).reshape(-1, 1)
        V += g @ g.T
    return V / N

def empirical_hessian_mean(theta_hat, X, Y, per_sample_loss_fn, per_sample_kwargs):
    """
    J = E[ ∇^2_θ ℓ_i(θ) ]
    """
    N, D = X.shape
    h_fn = hessian(lambda th, yi, xi: per_sample_loss_fn(th, yi, xi, **per_sample_kwargs))
    Hs = np.zeros((N, D, D))
    for i in range(N):
        Hs[i] = h_fn(theta_hat, Y[i], X[i])
    J = np.mean(Hs, axis=0)
    return Hs, J


# ============================================================
# C_raw helper (your "exact" discrete guidance piece)
# ============================================================
def compute_C_raw(hessians_samples, V, scaled_sandwich):
    """
    Matches your beta code structure:
      C_raw = E[ H_i Σ H_i ] + V - V/N
    where Σ is the target scaled sandwich covariance, and H_i is per-sample Hessian.
    """
    N = hessians_samples.shape[0]
    D = hessians_samples.shape[1]
    acc = np.zeros((D, D))
    for i in range(N):
        Hi = hessians_samples[i]
        acc += Hi @ scaled_sandwich @ Hi
    acc /= N
    acc += V
    acc -= V / N
    return acc


# ============================================================
# Generic SGD path generator (mini-batch)
# ============================================================
def sgd_path(loss_fn, theta_0, X, Y, n_iters, batch_size, lr_0,
             pre_matrix=None, precondition=True, fixed_lr=True,
             **loss_kwargs):
    """
    Returns array of shape (n_iters+1, d).
    """
    N, D = X.shape
    theta = np.copy(theta_0)
    path = np.zeros((n_iters + 1, D))
    path[0] = theta

    if (pre_matrix is None) or (not precondition):
        P = np.eye(D)
    else:
        P = np.asarray(pre_matrix)

    loss_grad = grad(lambda th, XX, YY: loss_fn(th, XX, YY, **loss_kwargs))

    for t in range(n_iters):
        gamma = lr_0 if fixed_lr else lr_0 / (t + 1)

        idx = np.random.randint(0, N, size=batch_size)
        Xb = X[idx]
        Yb = Y[idx]

        g = loss_grad(theta, Xb, Yb)  # gradient of mean loss on batch
        theta = theta - P @ (gamma * g)
        path[t + 1] = theta

    return path


# ============================================================
# Metrics
# ============================================================
def quantile_calibration(samples, theta_true):
    S = np.asarray(samples)
    tt = np.asarray(theta_true)
    return np.mean(S <= tt[None, :], axis=0)

def ks_to_uniform(qs):
    qs = np.sort(np.asarray(qs))
    m = qs.size
    grid = np.arange(1, m + 1) / m
    return np.max(np.abs(qs - grid))

def predictive_nll_from_samples(theta_samples, X_test, Y_test, sigma=1.0, max_samples=1000, seed=0):
    rng = np.random.default_rng(seed)
    S = theta_samples.shape[0]
    idx = rng.choice(S, size=min(S, max_samples), replace=False)
    thetas = np.asarray(theta_samples[idx])  # (S', D)

    mu = np.asarray(X_test) @ thetas.T
    logp = norm.logpdf(np.asarray(Y_test)[:, None], loc=mu, scale=sigma)
    log_pred = logsumexp(logp, axis=1) - np.log(logp.shape[1])
    return -np.mean(log_pred)

def summarize_over_batch(b_list, paths_by_b, theta_target, cov_target, X_test, Y_test,
                        burnin_frac=0.5, sigma_eval=1.0, max_samples=1000, seed=0):
    """
    Returns arrays (len(b),) for param_err, frob_cov, pred_nll, calib_ks.
    """
    theta_target = np.asarray(theta_target)
    cov_target = np.asarray(cov_target)
    theta_norm = np.linalg.norm(theta_target) + 1e-12

    out = {
        "param_err": np.zeros(len(b_list)),
        "frob_cov":  np.zeros(len(b_list)),
        "pred_nll":  np.zeros(len(b_list)),
        "calib_ks":  np.zeros(len(b_list)),
    }

    for i, B in enumerate(b_list):
        path = np.asarray(paths_by_b[i])
        T = path.shape[0]
        start = int(burnin_frac * T)
        S = path[start:]

        mean = S.mean(axis=0)
        emp_cov = np.cov(S, rowvar=False, bias=True)

        out["param_err"][i] = np.linalg.norm(mean - theta_target) / theta_norm
        out["frob_cov"][i]  = np.linalg.norm(emp_cov - cov_target, ord="fro")
        out["pred_nll"][i]  = predictive_nll_from_samples(S, X_test, Y_test, sigma=sigma_eval,
                                                          max_samples=max_samples, seed=seed)
        qs = quantile_calibration(S, theta_target)
        out["calib_ks"][i] = ks_to_uniform(qs)

    return out


# ============================================================
# Reference: Bayes posterior (flat prior, known sigma) and Gaussian sandwich
# ============================================================
def make_bayes_posterior_samples(X, Y, sigma=1.0, n_samples=5000, seed=0):
    rng = np.random.default_rng(seed)
    XtX = np.asarray(X.T @ X)
    XtY = np.asarray(X.T @ Y)
    theta_mle = np.linalg.solve(XtX, XtY)
    Sigma = (sigma**2) * np.linalg.inv(XtX)
    return rng.multivariate_normal(theta_mle, Sigma, size=n_samples), theta_mle, Sigma

def make_gaussian_samples(mean, cov, n_samples=5000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.asarray(mean), np.asarray(cov), size=n_samples)



import numpy as np
import pandas as pd

def run_one_rep(rep_seed: int,
                dimension=50,
                sample_size=2000,
                beta=1.5,
                sigma_working=1.0,
                n_mc=200,
                num_epochs=100,
                b_list=(16, 600),
                outlier_frac=0.1,
                outlier_scale=5.0,
                outlier_shift=10.0):
    """
    One Monte Carlo replication:
      - generate train/test
      - fit beta + nll
      - run 6 SGD methods
      - compute metrics for 8 methods
      - return long df with columns:
          rep, batch_size, method, param_err, frob_cov, pred_nll, calib_ks
    """
    # ---- IMPORTANT: control ALL randomness inside this rep
    np.random.seed(rep_seed)

    # ------------------------
    # Data
    # ------------------------
    model = NonlinearDependentNoise(N=sample_size, p=dimension, rho=0.0, bias=1.0, seed=rep_seed + 1000)
    X, Y = model.generate_data(outlier_frac=outlier_frac,
                               outlier_scale=outlier_scale,
                               outlier_shift=outlier_shift,
                               seed=rep_seed + 2000)

    theta_true = np.asarray(model.true_theta)

    # test set
    oldN = model.N
    model.N = 3000
    X_test, Y_test = model.generate_data(outlier_frac=outlier_frac,
                                         outlier_scale=outlier_scale,
                                         outlier_shift=outlier_shift,
                                         seed=rep_seed + 3000)
    model.N = oldN

    # ------------------------
    # (A) beta fit
    # ------------------------
    theta_init = np.zeros(dimension)
    theta_hat_beta = rmsprop(beta_loss, theta_init, X, Y,
                             lr=0.1, n_iter=120, print_every=None,
                             beta=beta, sigma=sigma_working, n_mc=n_mc)

    beta_kwargs = dict(beta=beta, sigma=sigma_working, n_mc=n_mc)
    V_beta = empirical_fisher_from_per_sample_grad(theta_hat_beta, X, Y, beta_loss_per_sample, beta_kwargs)
    H_beta_samples, J_beta = empirical_hessian_mean(theta_hat_beta, X, Y, beta_loss_per_sample, beta_kwargs)
    Jinv_beta = np.linalg.inv(J_beta)
    sandwich_beta = Jinv_beta @ V_beta @ Jinv_beta
    scaled_sandwich_beta = sandwich_beta / sample_size
    C_raw_beta = compute_C_raw(H_beta_samples, V_beta, scaled_sandwich_beta)

    # ------------------------
    # (B) nll fit (OLS)
    # ------------------------
    XtX = np.asarray(X.T @ X)
    XtY = np.asarray(X.T @ Y)
    theta_hat_nll = np.linalg.solve(XtX, XtY)

    nll_kwargs = dict(sigma=sigma_working)
    V_nll = empirical_fisher_from_per_sample_grad(theta_hat_nll, X, Y, nll_loss_per_sample, nll_kwargs)
    H_nll_samples, J_nll = empirical_hessian_mean(theta_hat_nll, X, Y, nll_loss_per_sample, nll_kwargs)
    Jinv_nll = np.linalg.inv(J_nll)
    sandwich_nll = Jinv_nll @ V_nll @ Jinv_nll
    scaled_sandwich_nll = sandwich_nll / sample_size
    C_raw_nll = compute_C_raw(H_nll_samples, V_nll, scaled_sandwich_nll)

    # ------------------------
    # Preconditioners + SGD
    # ------------------------
    pre_beta = build_preconds(J_beta, Jinv_beta, V_beta, scaled_sandwich_beta, C_raw_beta, X, Y, list(b_list))
    pre_nll  = build_preconds(J_nll,  Jinv_nll,  V_nll,  scaled_sandwich_nll,  C_raw_nll,  X, Y, list(b_list))

    beta_paths = run_three_sgd_methods(beta_loss,
                                       dict(beta=beta, sigma=sigma_working, n_mc=n_mc),
                                       theta0=theta_true,
                                       preconds=pre_beta,
                                       b_list=list(b_list), num_epochs=num_epochs, X=X, Y=Y)

    nll_paths  = run_three_sgd_methods(nll_loss,
                                       dict(sigma=sigma_working),
                                       theta0=theta_true,
                                       preconds=pre_nll,
                                       b_list=list(b_list), num_epochs=num_epochs, X=X, Y=Y)

    # ------------------------
    # References
    # ------------------------
    bayes_samples, _, _ = make_bayes_posterior_samples(X, Y, sigma=sigma_working, n_samples=5000, seed=rep_seed + 4000)
    sandwich_beta_samples = make_gaussian_samples(mean=theta_hat_beta, cov=scaled_sandwich_beta,
                                                  n_samples=5000, seed=rep_seed + 5000)

    # ------------------------
    # Metrics
    # ------------------------
    burnin_frac = 0.5
    sigma_eval = sigma_working

    beta_CT = summarize_over_batch(list(b_list), beta_paths["CT"], theta_true, scaled_sandwich_beta, X_test, Y_test,
                                   burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    beta_LS = summarize_over_batch(list(b_list), beta_paths["LS"], theta_true, scaled_sandwich_beta, X_test, Y_test,
                                   burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    beta_EX = summarize_over_batch(list(b_list), beta_paths["EX"], theta_true, scaled_sandwich_beta, X_test, Y_test,
                                   burnin_frac=burnin_frac, sigma_eval=sigma_eval)

    nll_CT = summarize_over_batch(list(b_list), nll_paths["CT"], theta_true, scaled_sandwich_nll, X_test, Y_test,
                                  burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    nll_LS = summarize_over_batch(list(b_list), nll_paths["LS"], theta_true, scaled_sandwich_nll, X_test, Y_test,
                                  burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    nll_EX = summarize_over_batch(list(b_list), nll_paths["EX"], theta_true, scaled_sandwich_nll, X_test, Y_test,
                                  burnin_frac=burnin_frac, sigma_eval=sigma_eval)

    def metrics_for_reference(samples, theta_target, cov_target):
        mean = samples.mean(axis=0)
        emp_cov = np.cov(samples, rowvar=False, bias=True)
        pe = np.linalg.norm(mean - theta_target) / (np.linalg.norm(theta_target) + 1e-12)
        frob = np.linalg.norm(emp_cov - cov_target, ord="fro")
        nllv = predictive_nll_from_samples(samples, X_test, Y_test, sigma=sigma_eval, max_samples=1000, seed=rep_seed)
        ks = ks_to_uniform(quantile_calibration(samples, theta_target))
        return pe, frob, nllv, ks

    pe_bayes, frob_bayes, nll_bayes, ks_bayes = metrics_for_reference(bayes_samples, theta_true, scaled_sandwich_nll)
    pe_sandB, frob_sandB, nll_sandB, ks_sandB = metrics_for_reference(sandwich_beta_samples, theta_true, scaled_sandwich_beta)

    # ------------------------
    # Build long df for this rep
    # ------------------------
    methods = [
        "continuous-time (log loss)",
        "large-sample+well-specified (log loss)",
        "exact (this paper, log loss)",
        r"continuous-time ($\beta = 1.5$)",
        r"large-sample+well-specified ($\beta = 1.5$)",
        r"exact (this paper, ($\beta = 1.5$))",
        r"Sandwich Gaussian ($\beta = 1.5$)",
        "Standard posterior",
    ]

    series = {
        "continuous-time (log loss)": dict(param_err=nll_CT["param_err"], frob_cov=nll_CT["frob_cov"],
                                           pred_nll=nll_CT["pred_nll"], calib_ks=nll_CT["calib_ks"]),
        "large-sample+well-specified (log loss)": dict(param_err=nll_LS["param_err"], frob_cov=nll_LS["frob_cov"],
                                                      pred_nll=nll_LS["pred_nll"], calib_ks=nll_LS["calib_ks"]),
        "exact (this paper, log loss)": dict(param_err=nll_EX["param_err"], frob_cov=nll_EX["frob_cov"],
                                             pred_nll=nll_EX["pred_nll"], calib_ks=nll_EX["calib_ks"]),
        r"continuous-time ($\beta = 1.5$)": dict(param_err=beta_CT["param_err"], frob_cov=beta_CT["frob_cov"],
                                                 pred_nll=beta_CT["pred_nll"], calib_ks=beta_CT["calib_ks"]),
        r"large-sample+well-specified ($\beta = 1.5$)": dict(param_err=beta_LS["param_err"], frob_cov=beta_LS["frob_cov"],
                                                            pred_nll=beta_LS["pred_nll"], calib_ks=beta_LS["calib_ks"]),
        r"exact (this paper, ($\beta = 1.5$))": dict(param_err=beta_EX["param_err"], frob_cov=beta_EX["frob_cov"],
                                                     pred_nll=beta_EX["pred_nll"], calib_ks=beta_EX["calib_ks"]),
        r"Sandwich Gaussian ($\beta = 1.5$)": dict(param_err=np.full(len(b_list), pe_sandB),
                                                   frob_cov=np.full(len(b_list), frob_sandB),
                                                   pred_nll=np.full(len(b_list), nll_sandB),
                                                   calib_ks=np.full(len(b_list), ks_sandB)),
        "Standard posterior": dict(param_err=np.full(len(b_list), pe_bayes),
                                   frob_cov=np.full(len(b_list), frob_bayes),
                                   pred_nll=np.full(len(b_list), nll_bayes),
                                   calib_ks=np.full(len(b_list), ks_bayes)),
    }

    rows = []
    for i, B in enumerate(list(b_list)):
        for m in methods:
            rows.append({
                "rep": rep_seed,
                "batch_size": int(B),
                "method": m,
                "param_err": float(series[m]["param_err"][i]),
                "frob_cov":  float(series[m]["frob_cov"][i]),
                "pred_nll":  float(series[m]["pred_nll"][i]),
                "calib_ks":  float(series[m]["calib_ks"][i]),
            })
    return pd.DataFrame(rows)

from math import sqrt

def ci_t_interval(x, alpha=0.05):
    """
    1D array -> (mean, lo, hi) using t-interval.
    Uses normal approx if df is large; for small df you can swap in scipy.stats.t.ppf if you want.
    """
    x = np.asarray(x, float)
    n = x.size
    m = x.mean()
    s = x.std(ddof=1) if n > 1 else 0.0
    se = s / sqrt(n) if n > 1 else 0.0

    # If you have scipy available, use: from scipy.stats import t; crit = t.ppf(1-alpha/2, df=n-1)
    # Fallback: normal 1.96 (fine once n is moderately large, e.g. >= 30)
    crit = 1.96 if n >= 30 else 2.262  # approx t_0.975, df=9 (reasonable if you use ~10 reps)
    return m, m - crit * se, m + crit * se

def ci_bootstrap(x, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    boot = []
    for _ in range(B):
        samp = rng.choice(x, size=n, replace=True)
        boot.append(samp.mean())
    boot = np.sort(np.asarray(boot))
    lo = np.quantile(boot, alpha/2)
    hi = np.quantile(boot, 1 - alpha/2)
    return x.mean(), lo, hi

def summarize_with_cis(df_all, metric, ci="t", alpha=0.05):
    """
    Returns a tidy df with mean/lo/hi for each (batch_size, method).
    """
    out_rows = []
    for (B, m), g in df_all.groupby(["batch_size", "method"]):
        vals = g[metric].to_numpy()
        if ci == "bootstrap":
            mean, lo, hi = ci_bootstrap(vals, B=2000, alpha=alpha, seed=123)
        else:
            mean, lo, hi = ci_t_interval(vals, alpha=alpha)
        out_rows.append({"batch_size": B, "method": m, "mean": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(out_rows)

def pivot_ci_table(df_ci, fmt="mean_pm"):
    """
    Pivot to batch_size x method with pretty strings.
      fmt="mean_pm":  mean ± halfwidth
      fmt="mean_ci":  mean [lo, hi]
    """
    df_ci = df_ci.copy()
    if fmt == "mean_ci":
        df_ci["cell"] = df_ci.apply(lambda r: f"{r['mean']:.4g} [{r['lo']:.4g}, {r['hi']:.4g}]", axis=1)
    else:
        hw = 0.5 * (df_ci["hi"] - df_ci["lo"])
        df_ci["cell"] = [f"{m:.4g} ± {h:.3g}" for m, h in zip(df_ci["mean"], hw)]
    return df_ci.pivot(index="batch_size", columns="method", values="cell").sort_index()


# =========================
# Run multiple reps
# =========================
R = 30 
rep_seeds = list(range(100, 100 + R))

dfs = []
for s in rep_seeds:
    dfs.append(run_one_rep(
        rep_seed=s,
        dimension=50,
        sample_size=2000,
        beta=1.5,
        sigma_working=1.0,
        n_mc=200,
        num_epochs=100,
        b_list=(16, int(0.3*2000)),
        outlier_frac=0.1,
        outlier_scale=5.0,
        outlier_shift=10.0
    ))
df_all = pd.concat(dfs, ignore_index=True)

# =========================
# Build CI tables
# =========================
for metric in ["param_err", "frob_cov", "pred_nll", "calib_ks"]:
    df_ci = summarize_with_cis(df_all, metric, ci="t", alpha=0.05)  # or ci="bootstrap"
    tab = pivot_ci_table(df_ci, fmt="mean_ci")  # or fmt="mean_pm"
    print("\n" + "="*120)
    print(f"{metric} : mean and 95% CI over {R} replications")
    print("="*120)
    print(tab.to_string())



# ============================================================
# Main experiment
# ============================================================
if __name__ == "__main__":
    # ------------------------
    # Config
    # ------------------------
    np.random.seed(100)
    dimension = 50
    sample_size = 2000

    beta = 1.5
    sigma_working = 1.0   # used in beta loss + NLL scoring model
    n_mc = 200

    num_epochs = 100
    b_list = [16, int(0.1*sample_size)] 

    # ------------------------
    # Data
    # ------------------------
    model = NonlinearDependentNoise(N=sample_size, p=dimension, rho=0.0, bias=1.0, seed=123)
    X, Y = model.generate_data(outlier_frac=0.1, outlier_scale=5.0, outlier_shift=10.0, seed=100)

    theta_true = np.asarray(model.true_theta)

    # test set for predictive NLL (use same contamination, or set outlier_frac=0 for clean test)
    oldN = model.N
    model.N = 3000
    X_test, Y_test = model.generate_data(outlier_frac=0.1, outlier_scale=5.0, outlier_shift=10.0, seed=2026)
    model.N = oldN

    # ============================================================
    # (A) beta-divergence fit: theta_hat_beta, J_beta, V_beta, sandwich_beta, C_raw_beta
    # ============================================================
    print("\n=== Fit beta-divergence (RMSProp) ===")
    theta_init = np.zeros(dimension)
    theta_hat_beta = rmsprop(beta_loss, theta_init, X, Y,
                             lr=0.1, n_iter=120, print_every=40,
                             beta=beta, sigma=sigma_working, n_mc=n_mc)

    # J_beta and V_beta
    beta_kwargs = dict(beta=beta, sigma=sigma_working, n_mc=n_mc)
    V_beta = empirical_fisher_from_per_sample_grad(
        theta_hat_beta, X, Y, beta_loss_per_sample, beta_kwargs
    )
    H_beta_samples, J_beta = empirical_hessian_mean(
        theta_hat_beta, X, Y, beta_loss_per_sample, beta_kwargs
    )
    Jinv_beta = np.linalg.inv(J_beta)

    sandwich_beta = Jinv_beta @ V_beta @ Jinv_beta
    scaled_sandwich_beta = sandwich_beta / sample_size

    C_raw_beta = compute_C_raw(H_beta_samples, V_beta, scaled_sandwich_beta)

    # ============================================================
    # (B) NLL fit: theta_hat_nll (OLS), J_nll, V_nll, sandwich_nll, C_raw_nll
    # ============================================================
    print("\n=== Fit NLL (Gaussian) ===")
    # OLS = MLE under homoskedastic Gaussian, sigma fixed doesn't affect argmin
    XtX = np.asarray(X.T @ X)
    XtY = np.asarray(X.T @ Y)
    theta_hat_nll = np.linalg.solve(XtX, XtY)

    nll_kwargs = dict(sigma=sigma_working)
    V_nll = empirical_fisher_from_per_sample_grad(
        theta_hat_nll, X, Y, nll_loss_per_sample, nll_kwargs
    )
    H_nll_samples, J_nll = empirical_hessian_mean(
        theta_hat_nll, X, Y, nll_loss_per_sample, nll_kwargs
    )
    Jinv_nll = np.linalg.inv(J_nll)

    sandwich_nll = Jinv_nll @ V_nll @ Jinv_nll
    scaled_sandwich_nll = sandwich_nll / sample_size

    C_raw_nll = compute_C_raw(H_nll_samples, V_nll, scaled_sandwich_nll)

    # ============================================================
    # Preconditioners for each tuning method (beta and nll)
    # ============================================================
    def build_preconds(J, Jinv, V, scaled_sandwich, C_raw, X, Y, b_list):
        """
        Returns dict with keys: 'CT', 'LS', 'EX'
        each value is dict mapping i -> preconditioner matrix for batch index i.
        """
        N, d = X.shape
        A = (X.T @ X) / N
        resid = Y - X @ np.linalg.solve(X.T @ X, X.T @ Y)
        variance_noise = np.var(resid)

        pre = {"CT": {}, "LS": {}, "EX": {}}

        for i, B in enumerate(b_list):
            # CT: use J^{-1}
            pre["CT"][i] = Jinv

            # LS+well-specified: your heuristic C formula, using target_cov = scaled_sandwich
            target_cov = scaled_sandwich
            C_ls = (A @ target_cov @ A + np.trace(A @ target_cov) * A) + (1 - d / N) * variance_noise * A
            C_ls = C_ls / B

            Lambda_ls = (1 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(C_ls + (1 / N) * V)
            pre["LS"][i] = Lambda_ls

            # EX: use your C_raw / B
            C_ex = C_raw / B
            Lambda_ex = (1 / N) * (V @ Jinv + Jinv @ V) @ np.linalg.inv(C_ex + (1 / N) * V)
            pre["EX"][i] = Lambda_ex

        return pre

    pre_beta = build_preconds(J_beta, Jinv_beta, V_beta, scaled_sandwich_beta, C_raw_beta, X, Y, b_list)
    pre_nll  = build_preconds(J_nll,  Jinv_nll,  V_nll,  scaled_sandwich_nll,  C_raw_nll,  X, Y, b_list)

    # ============================================================
    # Run SGD paths for 6 methods (beta: 3, nll: 3)
    # ============================================================
    def run_three_sgd_methods(loss_fn, loss_kwargs, theta0, preconds, b_list, num_epochs, X, Y):
        paths = {"CT": {}, "LS": {}, "EX": {}}
        N = X.shape[0]

        # your lr schedule: lr = 2*B/N for CT; you used lr_0=1 for the others
        lr_ct = [2 * B / N for B in b_list]

        for i, B in enumerate(b_list):
            n_iters = int(num_epochs * N / B)

            # CT
            paths["CT"][i] = sgd_path(loss_fn, theta0, X, Y,
                                     n_iters=n_iters, batch_size=B, lr_0=lr_ct[i],
                                     pre_matrix=preconds["CT"][i], precondition=True, fixed_lr=True,
                                     **loss_kwargs)

            # LS
            paths["LS"][i] = sgd_path(loss_fn, theta0, X, Y,
                                     n_iters=n_iters, batch_size=B, lr_0=1.0,
                                     pre_matrix=preconds["LS"][i], precondition=True, fixed_lr=True,
                                     **loss_kwargs)

            # EX
            paths["EX"][i] = sgd_path(loss_fn, theta0, X, Y,
                                     n_iters=n_iters, batch_size=B, lr_0=1.0,
                                     pre_matrix=preconds["EX"][i], precondition=True, fixed_lr=True,
                                     **loss_kwargs)
        return paths

    print("\n=== Run beta SGD methods ===")
    beta_paths = run_three_sgd_methods(beta_loss,
                                       dict(beta=beta, sigma=sigma_working, n_mc=n_mc),
                                       theta0=theta_true,   # match your previous choice
                                       preconds=pre_beta,
                                       b_list=b_list, num_epochs=num_epochs, X=X, Y=Y)

    print("\n=== Run NLL SGD methods ===")
    nll_paths = run_three_sgd_methods(nll_loss,
                                      dict(sigma=sigma_working),
                                      theta0=theta_true,
                                      preconds=pre_nll,
                                      b_list=b_list, num_epochs=num_epochs, X=X, Y=Y)

    # ============================================================
    # References: Bayes posterior (NLL) and Gaussian sandwich (beta)
    # ============================================================
    bayes_samples, theta_mle_ref, Sigma_bayes_ref = make_bayes_posterior_samples(X, Y, sigma=sigma_working,
                                                                                n_samples=5000, seed=0)
    sandwich_beta_samples = make_gaussian_samples(mean=theta_hat_beta, cov=scaled_sandwich_beta,
                                                  n_samples=5000, seed=0)

    # ============================================================
    # Evaluate all 8 methods
    #   - For Frobenius covariance metric, each method is compared to its own target:
    #       beta methods -> scaled_sandwich_beta
    #       nll methods  -> scaled_sandwich_nll
    #       Bayes (NLL)  -> scaled_sandwich_nll (same likelihood family)
    #       Sandwich(beta) -> scaled_sandwich_beta (by construction)
    #   - theta_target for param + calibration: use true_theta (simulation)
    # ============================================================
    print("\n=== Evaluate metrics ===")
    burnin_frac = 0.5
    sigma_eval = sigma_working

    # Beta methods
    beta_CT = summarize_over_batch(b_list, beta_paths["CT"], theta_true, scaled_sandwich_beta, X_test, Y_test,
                                   burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    beta_LS = summarize_over_batch(b_list, beta_paths["LS"], theta_true, scaled_sandwich_beta, X_test, Y_test,
                                   burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    beta_EX = summarize_over_batch(b_list, beta_paths["EX"], theta_true, scaled_sandwich_beta, X_test, Y_test,
                                   burnin_frac=burnin_frac, sigma_eval=sigma_eval)

    # NLL methods
    nll_CT = summarize_over_batch(b_list, nll_paths["CT"], theta_true, scaled_sandwich_nll, X_test, Y_test,
                                  burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    nll_LS = summarize_over_batch(b_list, nll_paths["LS"], theta_true, scaled_sandwich_nll, X_test, Y_test,
                                  burnin_frac=burnin_frac, sigma_eval=sigma_eval)
    nll_EX = summarize_over_batch(b_list, nll_paths["EX"], theta_true, scaled_sandwich_nll, X_test, Y_test,
                                  burnin_frac=burnin_frac, sigma_eval=sigma_eval)

    # Reference methods (single numbers)
    def metrics_for_reference(samples, theta_target, cov_target):
        mean = samples.mean(axis=0)
        emp_cov = np.cov(samples, rowvar=False, bias=True)
        pe = np.linalg.norm(mean - theta_target) / (np.linalg.norm(theta_target) + 1e-12)
        frob = np.linalg.norm(emp_cov - cov_target, ord="fro")
        nllv = predictive_nll_from_samples(samples, X_test, Y_test, sigma=sigma_eval, max_samples=1000, seed=0)
        ks = ks_to_uniform(quantile_calibration(samples, theta_target))
        return pe, frob, nllv, ks

    pe_bayes, frob_bayes, nll_bayes, ks_bayes = metrics_for_reference(bayes_samples, theta_true, scaled_sandwich_nll)
    pe_sandB, frob_sandB, nll_sandB, ks_sandB = metrics_for_reference(sandwich_beta_samples, theta_true, scaled_sandwich_beta)

    # ============================================================
    # Print table: per batch for 6 SGD methods + references
    # ============================================================
    
    # ---- Beta
    err_b_ct  = beta_CT["param_err"];  frob_b_ct = beta_CT["frob_cov"];  nll_b_ct  = beta_CT["pred_nll"];  ks_b_ct  = beta_CT["calib_ks"]
    err_b_ls  = beta_LS["param_err"];  frob_b_ls = beta_LS["frob_cov"];  nll_b_ls  = beta_LS["pred_nll"];  ks_b_ls  = beta_LS["calib_ks"]
    err_b_ex  = beta_EX["param_err"];  frob_b_ex = beta_EX["frob_cov"];  nll_b_ex  = beta_EX["pred_nll"];  ks_b_ex  = beta_EX["calib_ks"]
    
    # ---- NLL
    err_n_ct  = nll_CT["param_err"];   frob_n_ct = nll_CT["frob_cov"];   nll_n_ct  = nll_CT["pred_nll"];   ks_n_ct  = nll_CT["calib_ks"]
    err_n_ls  = nll_LS["param_err"];   frob_n_ls = nll_LS["frob_cov"];   nll_n_ls  = nll_LS["pred_nll"];   ks_n_ls  = nll_LS["calib_ks"]
    err_n_ex  = nll_EX["param_err"];   frob_n_ex = nll_EX["frob_cov"];   nll_n_ex  = nll_EX["pred_nll"];   ks_n_ex  = nll_EX["calib_ks"]

    
    header = (
        f"{'B':>7s} | "
        f"{'PE bCT':>9s} {'PE bLS':>9s} {'PE bEX':>9s} "
        f"{'PE nCT':>9s} {'PE nLS':>9s} {'PE nEX':>9s} "
        f"{'PE Bayes':>9s} {'PE Sandβ':>9s} | "
        f"{'F bCT':>9s} {'F bLS':>9s} {'F bEX':>9s} "
        f"{'F nCT':>9s} {'F nLS':>9s} {'F nEX':>9s} "
        f"{'F Bayes':>9s} {'F Sandβ':>9s} | "
        f"{'NLL bCT':>10s} {'NLL bLS':>10s} {'NLL bEX':>10s} "
        f"{'NLL nCT':>10s} {'NLL nLS':>10s} {'NLL nEX':>10s} "
        f"{'NLL Bay':>10s} {'NLL Sβ':>10s} | "
        f"{'KS bCT':>8s} {'KS bLS':>8s} {'KS bEX':>8s} "
        f"{'KS nCT':>8s} {'KS nLS':>8s} {'KS nEX':>8s} "
        f"{'KS Bay':>8s} {'KS Sβ':>8s}"
    )
    
    print(header)
    print("-" * len(header))
    
    
    for i, B in enumerate(b_list):
        print(
            f"{B:7d} | "
            f"{err_b_ct[i]:9.3g} {err_b_ls[i]:9.3g} {err_b_ex[i]:9.3g} "
            f"{err_n_ct[i]:9.3g} {err_n_ls[i]:9.3g} {err_n_ex[i]:9.3g} "
            f"{pe_bayes:9.3g} {pe_sandB:9.3g} | "
            f"{frob_b_ct[i]:9.3g} {frob_b_ls[i]:9.3g} {frob_b_ex[i]:9.3g} "
            f"{frob_n_ct[i]:9.3g} {frob_n_ls[i]:9.3g} {frob_n_ex[i]:9.3g} "
            f"{frob_bayes:9.3g} {frob_sandB:9.3g} | "
            f"{nll_b_ct[i]:10.3g} {nll_b_ls[i]:10.3g} {nll_b_ex[i]:10.3g} "
            f"{nll_n_ct[i]:10.3g} {nll_n_ls[i]:10.3g} {nll_n_ex[i]:10.3g} "
            f"{nll_bayes:10.3g} {nll_sandB:10.3g} | "
            f"{ks_b_ct[i]:8.3g} {ks_b_ls[i]:8.3g} {ks_b_ex[i]:8.3g} "
            f"{ks_n_ct[i]:8.3g} {ks_n_ls[i]:8.3g} {ks_n_ex[i]:8.3g} "
            f"{ks_bayes:8.3g} {ks_sandB:8.3g}"
        )
        

# import numpy as np
import matplotlib.pyplot as plt

b_arr = np.asarray(b_list)

labels = [
    "continuous-time (log loss)",
    "large-sample+well-specified (log loss)",
    "exact (this paper, log loss)",
    r"continuous-time ($\beta = 1.5$)",
    r"large-sample+well-specified ($\beta = 1.5$)",
    r"exact (this paper, ($\beta = 1.5$))",
    r"Sandwich Gaussian ($\beta = 1.5$)",
    "Standard posterior",
]

param_series = [
    np.asarray(err_n_ct),
    np.asarray(err_n_ls),
    np.asarray(err_n_ex),
    np.asarray(err_b_ct),
    np.asarray(err_b_ls),
    np.asarray(err_b_ex),
    np.full(len(b_arr), float(pe_sandB)),
    np.full(len(b_arr), float(pe_bayes)),
]

plt.figure(figsize=(6.8, 4.2))
for y, lab in zip(param_series, labels):
    plt.plot(b_arr, y, marker="o", linewidth=1.5, markersize=4, label=lab)

plt.xscale("log", base=2)
plt.xlabel("Batch size", fontsize=11)
plt.ylabel(r"$\|\hat{\theta}-\theta_{\star}\| \,/\, \|\theta_{\star}\|$", fontsize=11)
plt.title("Parameter Error", fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(True, alpha=0.25)

plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()


# import numpy as np
import matplotlib.pyplot as plt

b_arr = np.asarray(b_list)

labels = [
    "continuous-time (log loss)",
    "large-sample+well-specified (log loss)",
    "exact (this paper, log loss)",
    r"continuous-time ($\beta = 1.5$)",
    r"large-sample+well-specified ($\beta = 1.5$)",
    r"exact (this paper, ($\beta = 1.5$))",
    r"Sandwich Gaussian ($\beta = 1.5$)",
    "Standard posterior",
]

calib_series = [
    np.asarray(ks_n_ct),
    np.asarray(ks_n_ls),
    np.asarray(ks_n_ex),
    np.asarray(ks_b_ct),
    np.asarray(ks_b_ls),
    np.asarray(ks_b_ex),
    np.full(len(b_arr), float(ks_sandB)),
    np.full(len(b_arr), float(ks_bayes)),
]

plt.figure(figsize=(6.8, 4.2))
for y, lab in zip(calib_series, labels):
    plt.plot(b_arr, y, marker="o", linewidth=1.5, markersize=4, label=lab)

plt.xscale("log", base=2)
plt.xlabel("Batch size", fontsize=11)
plt.ylabel("KS error", fontsize=11)
plt.title("Quantile Calibration Error", fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(True, alpha=0.25)

plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()


# import numpy as np
import matplotlib.pyplot as plt

b_arr = np.asarray(b_list)

labels = [
    "continuous-time (log loss)",
    "large-sample+well-specified (log loss)",
    "exact (this paper, log loss)",
    r"continuous-time ($\beta = 1.5$)",
    r"large-sample+well-specified ($\beta = 1.5$)",
    r"exact (this paper, ($\beta = 1.5$))",
    "Sandwich Gaussian",
    "Standard posterior",
]

cov_series = [
    np.asarray(frob_n_ct),
    np.asarray(frob_n_ls),
    np.asarray(frob_n_ex),
    np.asarray(frob_b_ct),
    np.asarray(frob_b_ls),
    np.asarray(frob_b_ex),
    np.full(len(b_arr), float(frob_sandB)),
    np.full(len(b_arr), float(frob_bayes)),
]

plt.figure(figsize=(6.8, 4.2))
for y, lab in zip(cov_series, labels):
    plt.plot(b_arr, y, marker="o", linewidth=1.5, markersize=4, label=lab)

plt.xscale("log", base=2)
plt.xlabel("Batch size", fontsize=11)
plt.ylabel(r"$\|\mathcal{S}_{\star}-\hat{\mathcal{S}}\|_F$", fontsize=11)
plt.title("Covariance Error", fontsize=12)
plt.tick_params(labelsize=10)
plt.grid(True, alpha=0.25)

plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()


# =========================
# Consistent style settings
# =========================
style_map = {
    # Log loss (solid)
    "continuous-time (log loss)": dict(color="C0", ls="-", marker="o"),
    "large-sample+well-specified (log loss)": dict(color="C1", ls="--", marker="s"),
    "exact (this paper, log loss)": dict(color="C2", ls="-.", marker="^"),

    # Beta loss (dashed-dot family)
    r"continuous-time ($\beta = 1.5$)": dict(color="C0", ls=":", marker="o"),
    r"large-sample+well-specified ($\beta = 1.5$)": dict(color="C1", ls=":", marker="s"),
    r"exact (this paper, ($\beta = 1.5$))": dict(color="C2", ls=":", marker="^"),

    # References (thick horizontal)
    r"Sandwich Gaussian ($\beta = 1.5$)": dict(color="black", ls="--", lw=2.5),
    "Sandwich Gaussian": dict(color="black", ls="--", lw=2.5),
    "Standard posterior": dict(color="gray", ls="-", lw=2.5),
}


def plot_metric(b_arr, series, labels, ylabel, title):
    plt.figure(figsize=(6.8, 4.2))

    for y, lab in zip(series, labels):
        st = style_map[lab]
        plt.plot(
            b_arr, y,
            linewidth=st.get("lw", 1.6),
            linestyle=st.get("ls", "-"),
            marker=st.get("marker", None),
            markersize=4,
            color=st.get("color", None),
            label=lab,
        )

    plt.xscale("log", base=2)
    plt.xlabel("Batch size", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=12)
    plt.tick_params(labelsize=10)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()

plot_metric(
    b_arr,
    param_series,
    labels,
    ylabel=r"$\|\hat{\theta}-\theta_{\star}\| \,/\, \|\theta_{\star}\|$",
    title="Parameter Error",
)

plot_metric(
    b_arr,
    calib_series,
    labels,
    ylabel="KS error",
    title="Quantile Calibration Error",
)

plot_metric(
    b_arr,
    cov_series,
    labels,
    ylabel=r"$\|\mathcal{S}_{\star}-\hat{\mathcal{S}}\|_F$",
    title="Covariance Error",
)


# =========================
# TABLES INSTEAD OF PLOTS
# Paste after metrics are computed (after err_*, frob_*, nll_*, ks_* exist)
# =========================
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

def build_metric_arrays():
    """
    Collect all metric series into a consistent dict:
      series[method][metric] -> array of len(b_list) (or constant filled)
    """
    b_arr = np.asarray(b_list)

    series = {}

    # --- NLL / log-loss SGD methods
    series["continuous-time (log loss)"] = dict(
        param_err=np.asarray(err_n_ct),
        calib_ks=np.asarray(ks_n_ct),
        frob_cov=np.asarray(frob_n_ct),
        pred_nll=np.asarray(nll_n_ct),
    )
    series["large-sample+well-specified (log loss)"] = dict(
        param_err=np.asarray(err_n_ls),
        calib_ks=np.asarray(ks_n_ls),
        frob_cov=np.asarray(frob_n_ls),
        pred_nll=np.asarray(nll_n_ls),
    )
    series["exact (this paper, log loss)"] = dict(
        param_err=np.asarray(err_n_ex),
        calib_ks=np.asarray(ks_n_ex),
        frob_cov=np.asarray(frob_n_ex),
        pred_nll=np.asarray(nll_n_ex),
    )

    # --- beta-loss SGD methods
    series[r"continuous-time ($\beta = 1.5$)"] = dict(
        param_err=np.asarray(err_b_ct),
        calib_ks=np.asarray(ks_b_ct),
        frob_cov=np.asarray(frob_b_ct),
        pred_nll=np.asarray(nll_b_ct),
    )
    series[r"large-sample+well-specified ($\beta = 1.5$)"] = dict(
        param_err=np.asarray(err_b_ls),
        calib_ks=np.asarray(ks_b_ls),
        frob_cov=np.asarray(frob_b_ls),
        pred_nll=np.asarray(nll_b_ls),
    )
    series[r"exact (this paper, ($\beta = 1.5$))"] = dict(
        param_err=np.asarray(err_b_ex),
        calib_ks=np.asarray(ks_b_ex),
        frob_cov=np.asarray(frob_b_ex),
        pred_nll=np.asarray(nll_b_ex),
    )

    # --- references: constants across batch sizes
    series[r"Sandwich Gaussian ($\beta = 1.5$)"] = dict(
        param_err=np.full(len(b_arr), float(pe_sandB)),
        calib_ks=np.full(len(b_arr), float(ks_sandB)),
        frob_cov=np.full(len(b_arr), float(frob_sandB)),
        pred_nll=np.full(len(b_arr), float(nll_sandB)),
    )
    series["Standard posterior"] = dict(
        param_err=np.full(len(b_arr), float(pe_bayes)),
        calib_ks=np.full(len(b_arr), float(ks_bayes)),
        frob_cov=np.full(len(b_arr), float(frob_bayes)),
        pred_nll=np.full(len(b_arr), float(nll_bayes)),
    )

    return b_arr, series

def print_wide_table(metric_name, b_arr, series, floatfmt="{:.4g}"):
    """
    Wide layout: rows=batch size, cols=methods
    """
    methods = list(series.keys())

    header = ["B"] + methods
    print("\n" + "=" * 120)
    print(f"{metric_name} (rows=batch size, cols=methods)")
    print("=" * 120)
    print(" | ".join([f"{h:>18s}" for h in header]))
    print("-" * (21 * len(header)))

    for i, B in enumerate(b_arr):
        row = [f"{int(B):>18d}"]
        for m in methods:
            row.append(f"{floatfmt.format(series[m][metric_name][i]):>18s}")
        print(" | ".join(row))

def make_long_dataframe(b_arr, series):
    """
    Long/tidy layout: one row per (batch size, method)
    """
    if pd is None:
        raise ImportError("pandas is not installed. Install pandas or use print_wide_table().")

    rows = []
    for i, B in enumerate(b_arr):
        for method, mdict in series.items():
            rows.append({
                "batch_size": int(B),
                "method": method,
                "param_err": float(mdict["param_err"][i]),
                "frob_cov": float(mdict["frob_cov"][i]),
                "pred_nll": float(mdict["pred_nll"][i]),
                "calib_ks": float(mdict["calib_ks"][i]),
            })
    df = pd.DataFrame(rows)
    return df

def make_paper_table(df, metric, methods_order=None, floatfmt="{:.4g}"):
    """
    Pivot long df into a paper-friendly table:
      index=batch_size, columns=method, values=metric
    """
    if methods_order is not None:
        df = df.copy()
        df["method"] = pd.Categorical(df["method"], categories=methods_order, ordered=True)

    tab = df.pivot(index="batch_size", columns="method", values=metric).sort_index()
    # format
    return tab.applymap(lambda x: floatfmt.format(x))

# -------------------------
# Build and print tables
# -------------------------
b_arr, series = build_metric_arrays()

# 1) Print four wide tables (replaces your 3 plots; includes pred_nll too)
print_wide_table("param_err", b_arr, series)
print_wide_table("calib_ks",  b_arr, series)
print_wide_table("frob_cov",  b_arr, series)
print_wide_table("pred_nll",  b_arr, series)

# 2) If you have pandas, also create long + pivot tables + export
if pd is not None:
    df_long = make_long_dataframe(b_arr, series)
    print("\nLong/tidy table head:")
    print(df_long.head(12).to_string(index=False))

    # Choose an order matching your legend
    methods_order = [
        "continuous-time (log loss)",
        "large-sample+well-specified (log loss)",
        "exact (this paper, log loss)",
        r"continuous-time ($\beta = 1.5$)",
        r"large-sample+well-specified ($\beta = 1.5$)",
        r"exact (this paper, ($\beta = 1.5$))",
        r"Sandwich Gaussian ($\beta = 1.5$)",
        "Standard posterior",
    ]

    # Example: paper-ready pivot for param_err
    tab_param = make_paper_table(df_long, "param_err", methods_order=methods_order)
    print("\nPaper-style table (param_err):")
    print(tab_param.to_string())

    # Optional exports
    df_long.to_csv("results_long.csv", index=False)
    tab_param.to_latex("table_param_err.tex")

    # If you want a single LaTeX file with all metrics:
    for metric in ["param_err", "calib_ks", "frob_cov", "pred_nll"]:
        make_paper_table(df_long, metric, methods_order=methods_order).to_latex(f"table_{metric}.tex")
    print("\nSaved: results_long.csv, table_param_err.tex, and table_{metric}.tex for all metrics.")








