import autograd.numpy as np
from autograd import grad, hessian
from autograd.scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None


# ============================================================
# 0. Utilities
# ============================================================

def safe_inv(A, ridge=1e-8):
    A = np.asarray(A)
    d = A.shape[0]
    return np.linalg.inv(A + ridge * np.eye(d))


def symmetrize(A):
    return 0.5 * (A + A.T)


# ============================================================
# 1. Data generator
# ============================================================

class NonlinearDependentNoise:
    """
    Heteroskedastic linear data generator with optional response outliers.
    """

    def __init__(
        self,
        N=200,
        p=5,
        rho=0.0,
        penalty_para=0.0,
        theta_loc=0.0,
        theta_sigma=1.0,
        variance_list=(1, 2, 3),
        bias=1.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.p = int(p)
        self.N = int(N)
        self.rho = float(rho)
        self.penalty_para = float(penalty_para)
        self.bias = float(bias)

        self.true_theta = np.array([
            np.random.normal(theta_loc, theta_sigma)
            for _ in range(self.p)
        ])

        cov_theta = np.zeros((self.p, self.p))

        for i in range(self.p):
            cov_theta[i, i] = np.random.choice(variance_list)

        for i in range(self.p):
            for j in range(self.p):
                if i != j:
                    cov_theta[i, j] = (
                        rho
                        * np.sqrt(cov_theta[i, i])
                        * np.sqrt(cov_theta[j, j])
                    )

        self.cov_x = (
            np.linalg.inv(cov_theta)
            - penalty_para * np.eye(self.p)
        ) / self.N

    def generate_data(
        self,
        outlier_frac=0.0,
        outlier_scale=20.0,
        outlier_shift=0.0,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)

        mean = np.zeros(self.p)
        X = np.random.multivariate_normal(mean, self.cov_x, self.N)

        var_eps = np.array([
            1.0 + np.sum(X[i] ** 2)
            for i in range(self.N)
        ])

        mean_y = X.dot(self.true_theta) + self.bias
        Y = np.random.normal(mean_y, np.sqrt(var_eps))

        if outlier_frac > 0:
            n_out = int(outlier_frac * self.N)
            idx = np.random.choice(self.N, size=n_out, replace=False)

            Y[idx] = (
                outlier_shift
                + np.random.normal(
                    mean_y[idx],
                    outlier_scale * np.sqrt(var_eps[idx]),
                )
            )

        return X, Y


# ============================================================
# 2. Losses
# ============================================================

def f_density(y, x, theta, sigma=1.0):
    mu = np.dot(x, theta)
    return norm.pdf(y, mu, sigma)


def beta_loss(theta, X, Y, beta=1.5, sigma=1.0, n_mc=200):
    """
    Empirical beta-divergence objective using a Gaussian working model.
    """
    N, _ = X.shape
    z_samples = np.random.normal(0, sigma * 5, size=n_mc)

    total = 0.0

    for i in range(N):
        fy = f_density(Y[i], X[i], theta, sigma)

        mu_i = np.dot(X[i], theta)
        fz = norm.pdf(z_samples, mu_i, sigma)

        integral_term = np.mean(fz ** beta)

        total += (
            -fy ** (beta - 1) / (beta - 1)
            + integral_term / beta
        )

    return total / N


def beta_loss_per_sample(theta, y, x, beta=1.5, sigma=1.0, n_mc=200):
    fy = f_density(y, x, theta, sigma)

    z_samples = np.random.normal(0, sigma * 5, size=n_mc)
    fz = f_density(z_samples, x, theta, sigma)

    integral_term = np.mean(fz ** beta)

    return (
        -fy ** (beta - 1) / (beta - 1)
        + integral_term / beta
    )


def nll_loss(theta, X, Y, sigma=1.0):
    """
    Gaussian negative log-likelihood with fixed sigma.
    """
    r = (Y - X @ theta) / sigma

    return np.mean(
        0.5 * r ** 2
        + np.log(sigma)
        + 0.5 * np.log(2 * np.pi)
    )


def nll_loss_per_sample(theta, y, x, sigma=1.0):
    r = (y - np.dot(x, theta)) / sigma

    return (
        0.5 * r ** 2
        + np.log(sigma)
        + 0.5 * np.log(2 * np.pi)
    )


# ============================================================
# 3. Optimizer
# ============================================================

def rmsprop(
    loss_fn,
    theta_init,
    X,
    Y,
    lr=0.05,
    decay=0.9,
    eps=1e-8,
    n_iter=200,
    print_every=50,
    **loss_kwargs,
):
    theta = np.copy(theta_init)
    g_sq = np.zeros_like(theta)

    loss_grad = grad(
        lambda th, XX, YY: loss_fn(th, XX, YY, **loss_kwargs)
    )

    for t in range(n_iter):
        g = loss_grad(theta, X, Y)

        g_sq = decay * g_sq + (1 - decay) * (g ** 2)

        theta -= lr * g / (np.sqrt(g_sq) + eps)

        if print_every is not None and t % print_every == 0:
            L = loss_fn(theta, X, Y, **loss_kwargs)
            print(f"RMSProp iter {t:04d} | loss = {L:.6f}")

    return theta


# ============================================================
# 4. J and V
# ============================================================

def empirical_fisher_from_per_sample_grad(
    theta_hat,
    X,
    Y,
    per_sample_loss_fn,
    per_sample_kwargs,
):
    """
    V = E[g_i g_i^T], where g_i = grad ell_i(theta_hat).
    """
    N, D = X.shape

    g_fn = grad(
        lambda th, yi, xi: per_sample_loss_fn(
            th,
            yi,
            xi,
            **per_sample_kwargs,
        )
    )

    V = np.zeros((D, D))

    for i in range(N):
        g = g_fn(theta_hat, Y[i], X[i]).reshape(-1, 1)
        V += g @ g.T

    V /= N
    return symmetrize(V)


def empirical_hessian_mean(
    theta_hat,
    X,
    Y,
    per_sample_loss_fn,
    per_sample_kwargs,
):
    """
    J = E[ Hessian ell_i(theta_hat) ].
    """
    N, D = X.shape

    h_fn = hessian(
        lambda th, yi, xi: per_sample_loss_fn(
            th,
            yi,
            xi,
            **per_sample_kwargs,
        )
    )

    Hs = np.zeros((N, D, D))

    for i in range(N):
        Hs[i] = h_fn(theta_hat, Y[i], X[i])

    J = np.mean(Hs, axis=0)
    J = symmetrize(J)

    return Hs, J


def compute_C_raw(hessians_samples, V, scaled_sandwich):
    """
    C_raw = E[H_i Sigma H_i] + V - V/N.
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

    return symmetrize(acc)


# ============================================================
# 5. SGD path generator
# ============================================================

def sgd_path(
    loss_fn,
    theta_0,
    X,
    Y,
    n_iters,
    batch_size,
    lr_0,
    pre_matrix=None,
    precondition=True,
    fixed_lr=True,
    **loss_kwargs,
):
    """
    Generate an SGD path.
    """
    N, D = X.shape

    theta = np.copy(theta_0)

    path = np.zeros((n_iters + 1, D))
    path[0] = theta

    if pre_matrix is None or not precondition:
        P = np.eye(D)
    else:
        P = np.asarray(pre_matrix)

    loss_grad = grad(
        lambda th, XX, YY: loss_fn(th, XX, YY, **loss_kwargs)
    )

    for t in range(n_iters):
        gamma = lr_0 if fixed_lr else lr_0 / (t + 1)

        idx = np.random.randint(0, N, size=batch_size)
        Xb = X[idx]
        Yb = Y[idx]

        g = loss_grad(theta, Xb, Yb)

        theta = theta - P @ (gamma * g)

        path[t + 1] = theta

        if not np.isfinite(theta).all():
            path = path[: t + 2]
            break

    return path


# ============================================================
# 6. Metrics
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


def predictive_nll_from_samples(
    theta_samples,
    X_test,
    Y_test,
    sigma=1.0,
    max_samples=1000,
    seed=0,
):
    rng = np.random.default_rng(seed)

    S = theta_samples.shape[0]

    idx = rng.choice(
        S,
        size=min(S, max_samples),
        replace=False,
    )

    thetas = np.asarray(theta_samples[idx])

    mu = np.asarray(X_test) @ thetas.T

    logp = norm.logpdf(
        np.asarray(Y_test)[:, None],
        loc=mu,
        scale=sigma,
    )

    log_pred = logsumexp(logp, axis=1) - np.log(logp.shape[1])

    return -np.mean(log_pred)


def summarize_over_batch(
    b_list,
    paths_by_b,
    theta_target,
    cov_target,
    X_test,
    Y_test,
    burnin_frac=0.5,
    sigma_eval=1.0,
    max_samples=1000,
    seed=0,
):
    """
    Returns arrays over batch size for:
      param_err, frob_cov, pred_nll, calib_ks.

    frob_cov is the relative Frobenius covariance error:
      ||emp_cov - cov_target||_F / ||cov_target||_F.
    """
    theta_target = np.asarray(theta_target)
    cov_target = np.asarray(cov_target)

    theta_norm = np.linalg.norm(theta_target) + 1e-12
    cov_norm = np.linalg.norm(cov_target, ord="fro") + 1e-12

    out = {
        "param_err": np.zeros(len(b_list)),
        "frob_cov": np.zeros(len(b_list)),
        "pred_nll": np.zeros(len(b_list)),
        "calib_ks": np.zeros(len(b_list)),
    }

    for i, B in enumerate(b_list):
        path = np.asarray(paths_by_b[i])

        T = path.shape[0]
        start = int(burnin_frac * T)

        S = path[start:]

        if S.shape[0] < 5 or not np.isfinite(S).all():
            out["param_err"][i] = np.nan
            out["frob_cov"][i] = np.nan
            out["pred_nll"][i] = np.nan
            out["calib_ks"][i] = np.nan
            continue

        mean = S.mean(axis=0)
        emp_cov = np.cov(S, rowvar=False, bias=True)

        out["param_err"][i] = (
            np.linalg.norm(mean - theta_target) / theta_norm
        )

        out["frob_cov"][i] = (
            np.linalg.norm(emp_cov - cov_target, ord="fro") / cov_norm
        )

        out["pred_nll"][i] = predictive_nll_from_samples(
            S,
            X_test,
            Y_test,
            sigma=sigma_eval,
            max_samples=max_samples,
            seed=seed,
        )

        qs = quantile_calibration(S, theta_target)
        out["calib_ks"][i] = ks_to_uniform(qs)

    return out


# ============================================================
# 7. Reference distributions
# ============================================================

def make_bayes_posterior_samples(
    X,
    Y,
    sigma=1.0,
    n_samples=5000,
    seed=0,
):
    rng = np.random.default_rng(seed)

    XtX = np.asarray(X.T @ X)
    XtY = np.asarray(X.T @ Y)

    theta_mle = np.linalg.solve(XtX, XtY)
    Sigma = sigma ** 2 * np.linalg.inv(XtX)

    samples = rng.multivariate_normal(theta_mle, Sigma, size=n_samples)

    return samples, theta_mle, Sigma


def make_gaussian_samples(mean, cov, n_samples=5000, seed=0):
    rng = np.random.default_rng(seed)

    return rng.multivariate_normal(
        np.asarray(mean),
        np.asarray(cov),
        size=n_samples,
    )


# ============================================================
# 8. Preconditioners
# ============================================================

def build_preconds(
    J,
    Jinv,
    V,
    scaled_sandwich,
    C_raw,
    X,
    Y,
    b_list,
    ridge=1e-8,
):
    """
    Returns preconditioners for:
      CT, DQ+const, DQ+Exact.
    """
    N, d = X.shape

    A = (X.T @ X) / N

    theta_ols = np.linalg.solve(X.T @ X, X.T @ Y)
    resid = Y - X @ theta_ols
    variance_noise = np.var(resid)

    pre = {
        "CT": {},
        "DQ+const": {},
        "DQ+Exact": {},
    }

    for i, B in enumerate(b_list):
        # CT
        pre["CT"][i] = Jinv

        # DQ+const: large-sample + well-specified approximation
        target_cov = scaled_sandwich

        C_ls = (
            A @ target_cov @ A
            + np.trace(A @ target_cov) * A
            + (1 - d / N) * variance_noise * A
        )

        C_ls = C_ls / B

        denom_ls = C_ls + (1 / N) * V + ridge * np.eye(d)

        Lambda_ls = (
            (1 / N)
            * (V @ Jinv + Jinv @ V)
            @ np.linalg.inv(denom_ls)
        )

        pre["DQ+const"][i] = Lambda_ls

        # DQ+Exact
        C_ex = C_raw / B

        denom_ex = C_ex + (1 / N) * V + ridge * np.eye(d)

        Lambda_ex = (
            (1 / N)
            * (V @ Jinv + Jinv @ V)
            @ np.linalg.inv(denom_ex)
        )

        pre["DQ+Exact"][i] = Lambda_ex

    return pre


# ============================================================
# 9. Run three SGD methods
# ============================================================

def run_three_sgd_methods(
    loss_fn,
    loss_kwargs,
    theta0,
    preconds,
    b_list,
    num_epochs,
    X,
    Y,
):
    paths = {
        "CT": {},
        "DQ+const": {},
        "DQ+Exact": {},
    }

    N = X.shape[0]

    lr_ct = [2 * B / N for B in b_list]

    for i, B in enumerate(b_list):
        n_iters = int(num_epochs * N / B)

        paths["CT"][i] = sgd_path(
            loss_fn,
            theta0,
            X,
            Y,
            n_iters=n_iters,
            batch_size=B,
            lr_0=lr_ct[i],
            pre_matrix=preconds["CT"][i],
            precondition=True,
            fixed_lr=True,
            **loss_kwargs,
        )

        paths["DQ+const"][i] = sgd_path(
            loss_fn,
            theta0,
            X,
            Y,
            n_iters=n_iters,
            batch_size=B,
            lr_0=1.0,
            pre_matrix=preconds["DQ+const"][i],
            precondition=True,
            fixed_lr=True,
            **loss_kwargs,
        )

        paths["DQ+Exact"][i] = sgd_path(
            loss_fn,
            theta0,
            X,
            Y,
            n_iters=n_iters,
            batch_size=B,
            lr_0=1.0,
            pre_matrix=preconds["DQ+Exact"][i],
            precondition=True,
            fixed_lr=True,
            **loss_kwargs,
        )

    return paths


# ============================================================
# 10. One replication
# ============================================================

def run_one_rep(
    rep_seed: int,
    dimension=50,
    sample_size=2000,
    beta=1.5,
    sigma_working=1.0,
    n_mc=200,
    num_epochs=100,
    b_list=(16, 600),
    outlier_frac=0.1,
    outlier_scale=5.0,
    outlier_shift=10.0,
):
    """
    Returns a long dataframe for this replication.

    Columns:
      rep, loss, batch_size, method,
      param_err, frob_cov, pred_nll, calib_ks.
    """
    if pd is None:
        raise ImportError(
            "Please install pandas with: pip install pandas"
        )

    np.random.seed(rep_seed)

    # ------------------------
    # Data
    # ------------------------
    model = NonlinearDependentNoise(
        N=sample_size,
        p=dimension,
        rho=0.0,
        bias=1.0,
        seed=rep_seed + 123,
    )

    X, Y = model.generate_data(
        outlier_frac=outlier_frac,
        outlier_scale=outlier_scale,
        outlier_shift=outlier_shift,
        seed=rep_seed + 456,
    )

    theta_true = np.asarray(model.true_theta)

    # ------------------------
    # Test set
    # ------------------------
    oldN = model.N

    model.N = 3000

    X_test, Y_test = model.generate_data(
        outlier_frac=outlier_frac,
        outlier_scale=outlier_scale,
        outlier_shift=outlier_shift,
        seed=rep_seed + 789,
    )

    model.N = oldN

    # ============================================================
    # A. Beta fit
    # ============================================================

    theta_init = np.zeros(dimension)

    theta_hat_beta = rmsprop(
        beta_loss,
        theta_init,
        X,
        Y,
        lr=0.1,
        n_iter=120,
        print_every=None,
        beta=beta,
        sigma=sigma_working,
        n_mc=n_mc,
    )

    beta_kwargs = {
        "beta": beta,
        "sigma": sigma_working,
        "n_mc": n_mc,
    }

    V_beta = empirical_fisher_from_per_sample_grad(
        theta_hat_beta,
        X,
        Y,
        beta_loss_per_sample,
        beta_kwargs,
    )

    H_beta_samples, J_beta = empirical_hessian_mean(
        theta_hat_beta,
        X,
        Y,
        beta_loss_per_sample,
        beta_kwargs,
    )

    Jinv_beta = safe_inv(J_beta)

    sandwich_beta = Jinv_beta @ V_beta @ Jinv_beta
    sandwich_beta = symmetrize(sandwich_beta)

    scaled_sandwich_beta = sandwich_beta / sample_size
    scaled_sandwich_beta = symmetrize(scaled_sandwich_beta)

    C_raw_beta = compute_C_raw(
        H_beta_samples,
        V_beta,
        scaled_sandwich_beta,
    )

    # ============================================================
    # B. Log-loss / NLL fit
    # ============================================================

    XtX = np.asarray(X.T @ X)
    XtY = np.asarray(X.T @ Y)

    theta_hat_nll = np.linalg.solve(XtX, XtY)

    nll_kwargs = {
        "sigma": sigma_working,
    }

    V_nll = empirical_fisher_from_per_sample_grad(
        theta_hat_nll,
        X,
        Y,
        nll_loss_per_sample,
        nll_kwargs,
    )

    H_nll_samples, J_nll = empirical_hessian_mean(
        theta_hat_nll,
        X,
        Y,
        nll_loss_per_sample,
        nll_kwargs,
    )

    Jinv_nll = safe_inv(J_nll)

    sandwich_nll = Jinv_nll @ V_nll @ Jinv_nll
    sandwich_nll = symmetrize(sandwich_nll)

    scaled_sandwich_nll = sandwich_nll / sample_size
    scaled_sandwich_nll = symmetrize(scaled_sandwich_nll)

    C_raw_nll = compute_C_raw(
        H_nll_samples,
        V_nll,
        scaled_sandwich_nll,
    )

    # ============================================================
    # Preconditioners
    # ============================================================

    b_list = list(b_list)

    pre_beta = build_preconds(
        J_beta,
        Jinv_beta,
        V_beta,
        scaled_sandwich_beta,
        C_raw_beta,
        X,
        Y,
        b_list,
    )

    pre_nll = build_preconds(
        J_nll,
        Jinv_nll,
        V_nll,
        scaled_sandwich_nll,
        C_raw_nll,
        X,
        Y,
        b_list,
    )

    # ============================================================
    # Run SGD paths
    # ============================================================

    beta_paths = run_three_sgd_methods(
        beta_loss,
        beta_kwargs,
        theta0=theta_true,
        preconds=pre_beta,
        b_list=b_list,
        num_epochs=num_epochs,
        X=X,
        Y=Y,
    )

    nll_paths = run_three_sgd_methods(
        nll_loss,
        nll_kwargs,
        theta0=theta_true,
        preconds=pre_nll,
        b_list=b_list,
        num_epochs=num_epochs,
        X=X,
        Y=Y,
    )

    # ============================================================
    # Reference methods
    # ============================================================

    bayes_samples, _, _ = make_bayes_posterior_samples(
        X,
        Y,
        sigma=sigma_working,
        n_samples=5000,
        seed=rep_seed + 111,
    )

    sandwich_beta_samples = make_gaussian_samples(
        mean=theta_hat_beta,
        cov=scaled_sandwich_beta,
        n_samples=5000,
        seed=rep_seed + 222,
    )

    # ============================================================
    # Evaluate metrics
    # ============================================================

    burnin_frac = 0.5
    sigma_eval = sigma_working

    beta_summaries = {
        "CT": summarize_over_batch(
            b_list,
            beta_paths["CT"],
            theta_true,
            scaled_sandwich_beta,
            X_test,
            Y_test,
            burnin_frac=burnin_frac,
            sigma_eval=sigma_eval,
            seed=rep_seed,
        ),
        "DQ+const": summarize_over_batch(
            b_list,
            beta_paths["DQ+const"],
            theta_true,
            scaled_sandwich_beta,
            X_test,
            Y_test,
            burnin_frac=burnin_frac,
            sigma_eval=sigma_eval,
            seed=rep_seed,
        ),
        "DQ+Exact": summarize_over_batch(
            b_list,
            beta_paths["DQ+Exact"],
            theta_true,
            scaled_sandwich_beta,
            X_test,
            Y_test,
            burnin_frac=burnin_frac,
            sigma_eval=sigma_eval,
            seed=rep_seed,
        ),
    }

    nll_summaries = {
        "CT": summarize_over_batch(
            b_list,
            nll_paths["CT"],
            theta_true,
            scaled_sandwich_nll,
            X_test,
            Y_test,
            burnin_frac=burnin_frac,
            sigma_eval=sigma_eval,
            seed=rep_seed,
        ),
        "DQ+const": summarize_over_batch(
            b_list,
            nll_paths["DQ+const"],
            theta_true,
            scaled_sandwich_nll,
            X_test,
            Y_test,
            burnin_frac=burnin_frac,
            sigma_eval=sigma_eval,
            seed=rep_seed,
        ),
        "DQ+Exact": summarize_over_batch(
            b_list,
            nll_paths["DQ+Exact"],
            theta_true,
            scaled_sandwich_nll,
            X_test,
            Y_test,
            burnin_frac=burnin_frac,
            sigma_eval=sigma_eval,
            seed=rep_seed,
        ),
    }

    def metrics_for_reference(samples, theta_target, cov_target):
        mean = samples.mean(axis=0)
        emp_cov = np.cov(samples, rowvar=False, bias=True)

        pe = (
            np.linalg.norm(mean - theta_target)
            / (np.linalg.norm(theta_target) + 1e-12)
        )

        cov_norm = np.linalg.norm(cov_target, ord="fro") + 1e-12

        frob = (
            np.linalg.norm(emp_cov - cov_target, ord="fro")
            / cov_norm
        )

        nllv = predictive_nll_from_samples(
            samples,
            X_test,
            Y_test,
            sigma=sigma_eval,
            max_samples=1000,
            seed=0,
        )

        ks = ks_to_uniform(
            quantile_calibration(samples, theta_target)
        )

        return pe, frob, nllv, ks

    pe_bayes, frob_bayes, nll_bayes, ks_bayes = metrics_for_reference(
        bayes_samples,
        theta_true,
        scaled_sandwich_nll,
    )

    pe_sandB, frob_sandB, nll_sandB, ks_sandB = metrics_for_reference(
        sandwich_beta_samples,
        theta_true,
        scaled_sandwich_beta,
    )

    # ============================================================
    # Long dataframe
    # ============================================================

    rows = []

    base_methods = ["CT", "DQ+const", "DQ+Exact"]

    for i, B in enumerate(b_list):
        for method in base_methods:
            rows.append({
                "rep": int(rep_seed),
                "loss": "log-loss",
                "batch_size": int(B),
                "method": method,
                "param_err": float(nll_summaries[method]["param_err"][i]),
                "frob_cov": float(nll_summaries[method]["frob_cov"][i]),
                "pred_nll": float(nll_summaries[method]["pred_nll"][i]),
                "calib_ks": float(nll_summaries[method]["calib_ks"][i]),
            })

        for method in base_methods:
            rows.append({
                "rep": int(rep_seed),
                "loss": r"$\beta$-loss",
                "batch_size": int(B),
                "method": method,
                "param_err": float(beta_summaries[method]["param_err"][i]),
                "frob_cov": float(beta_summaries[method]["frob_cov"][i]),
                "pred_nll": float(beta_summaries[method]["pred_nll"][i]),
                "calib_ks": float(beta_summaries[method]["calib_ks"][i]),
            })

        rows.append({
            "rep": int(rep_seed),
            "loss": "reference",
            "batch_size": int(B),
            "method": "Standard posterior",
            "param_err": float(pe_bayes),
            "frob_cov": float(frob_bayes),
            "pred_nll": float(nll_bayes),
            "calib_ks": float(ks_bayes),
        })

        rows.append({
            "rep": int(rep_seed),
            "loss": "reference",
            "batch_size": int(B),
            "method": r"Sandwich Gaussian ($\beta$)",
            "param_err": float(pe_sandB),
            "frob_cov": float(frob_sandB),
            "pred_nll": float(nll_sandB),
            "calib_ks": float(ks_sandB),
        })

    return pd.DataFrame(rows)


# ============================================================
# 11. CI helpers
# ============================================================

def ci_normal(x, alpha=0.05):
    """
    Mean +/- z_{1-alpha/2} * standard error.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]

    n = x.size

    if n == 0:
        return np.nan, np.nan, np.nan

    m = float(x.mean())

    if n == 1:
        return m, m, m

    s = float(x.std(ddof=1))
    se = s / np.sqrt(n)

    z = 1.959963984540054

    return m, m - z * se, m + z * se


def ci_bootstrap_mean(x, B=2000, alpha=0.05, seed=0):
    """
    Percentile bootstrap CI for the mean.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]

    n = x.size

    if n == 0:
        return np.nan, np.nan, np.nan

    boot = []

    for _ in range(B):
        samp = rng.choice(x, size=n, replace=True)
        boot.append(float(samp.mean()))

    boot = np.sort(np.asarray(boot))

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))

    return float(x.mean()), lo, hi


def summarize_with_cis(df_all, metric, ci_method="normal", alpha=0.05):
    """
    Returns tidy dataframe:
      loss, batch_size, method, mean, lo, hi.
    """
    if pd is None:
        raise ImportError("pandas required for CI tables.")

    out_rows = []

    for (loss, B, method), g in df_all.groupby(["loss", "batch_size", "method"]):
        vals = g[metric].to_numpy()

        if ci_method == "bootstrap":
            mean, lo, hi = ci_bootstrap_mean(
                vals,
                B=2000,
                alpha=alpha,
                seed=123,
            )
        else:
            mean, lo, hi = ci_normal(vals, alpha=alpha)

        out_rows.append({
            "loss": loss,
            "batch_size": int(B),
            "method": method,
            "mean": mean,
            "lo": lo,
            "hi": hi,
        })

    return pd.DataFrame(out_rows)


def pivot_ci_table(df_ci, fmt="mean_ci", digits=4):
    """
    fmt:
      - "mean_ci": mean [lo, hi]
      - "mean_pm": mean ± halfwidth
    """
    df_ci = df_ci.copy()

    if fmt == "mean_pm":
        hw = 0.5 * (df_ci["hi"] - df_ci["lo"])

        df_ci["cell"] = [
            f"{m:.{digits}g} ± {h:.{digits}g}"
            for m, h in zip(df_ci["mean"], hw)
        ]
    else:
        df_ci["cell"] = df_ci.apply(
            lambda r: (
                f"{r['mean']:.{digits}g} "
                f"[{r['lo']:.{digits}g}, {r['hi']:.{digits}g}]"
            ),
            axis=1,
        )

    tab = (
        df_ci
        .pivot_table(
            index=["loss", "batch_size"],
            columns="method",
            values="cell",
            aggfunc="first",
        )
        .sort_index()
    )

    return tab


def pivot_mean_only_table(df_ci, digits=4):
    """
    Mean-only table, useful for main text.
    """
    df_ci = df_ci.copy()

    df_ci["cell"] = [
        f"{m:.{digits}g}" if np.isfinite(m) else "NA"
        for m in df_ci["mean"]
    ]

    tab = (
        df_ci
        .pivot_table(
            index=["loss", "batch_size"],
            columns="method",
            values="cell",
            aggfunc="first",
        )
        .sort_index()
    )

    return tab


# ============================================================
# 12. Main experiment
# ============================================================

if __name__ == "__main__":

    if pd is None:
        raise SystemExit(
            "This script needs pandas for CI tables. "
            "Install via: pip install pandas"
        )

    # ------------------------
    # Config
    # ------------------------

    DIMENSION = 20
    SAMPLE_SIZE = 5000

    BETA = 1.5
    SIGMA_WORKING = 1.0
    N_MC = 50

    NUM_EPOCHS = 30
    B_LIST = (16, int(0.1 * SAMPLE_SIZE))

    OUTLIER_FRAC = 0.1
    OUTLIER_SCALE = 5.0
    OUTLIER_SHIFT = 10.0

    # ---- Replications
    R = 30
    REP_SEEDS = list(range(100, 100 + R))

    # ---- CI method
    CI_METHOD = "bootstrap" if R < 30 else "normal"
    CI_FMT = "mean_ci"

    # Choose which metrics to report.
    # For main-text compact tables, use for example:
    # REPORT_METRICS = ["calib_ks", "frob_cov"]
    REPORT_METRICS = [
        "param_err",
        "frob_cov",
        "pred_nll",
        "calib_ks",
    ]

    METRIC_NAMES = {
        "param_err": "parameter_error",
        "frob_cov": "covariance_error",
        "pred_nll": "predictive_nll",
        "calib_ks": "calibration_error",
    }

    print(f"Running {R} replications | CI_METHOD={CI_METHOD}")

    # ------------------------
    # Run replications
    # ------------------------

    dfs = []

    for s in REP_SEEDS:
        print(f"  Rep seed {s}")

        df_rep = run_one_rep(
            rep_seed=s,
            dimension=DIMENSION,
            sample_size=SAMPLE_SIZE,
            beta=BETA,
            sigma_working=SIGMA_WORKING,
            n_mc=N_MC,
            num_epochs=NUM_EPOCHS,
            b_list=B_LIST,
            outlier_frac=OUTLIER_FRAC,
            outlier_scale=OUTLIER_SCALE,
            outlier_shift=OUTLIER_SHIFT,
        )

        dfs.append(df_rep)

    df_all = pd.concat(dfs, ignore_index=True)

    # ------------------------
    # Save raw results
    # ------------------------

    df_all.to_csv("results_long_with_reps.csv", index=False)
    print("Saved: results_long_with_reps.csv")

    # ------------------------
    # CI tables
    # ------------------------

    for metric in REPORT_METRICS:
        metric_name = METRIC_NAMES.get(metric, metric)

        df_ci = summarize_with_cis(
            df_all,
            metric,
            ci_method=CI_METHOD,
            alpha=0.05,
        )

        tab_ci = pivot_ci_table(
            df_ci,
            fmt=CI_FMT,
            digits=4,
        )

        tab_mean = pivot_mean_only_table(
            df_ci,
            digits=4,
        )

        print("\n" + "=" * 140)
        print(f"{metric_name}: mean and 95% CI over {R} reps")
        print("=" * 140)
        print(tab_ci.to_string())

        ci_csv = f"table_{metric_name}_ci.csv"
        mean_csv = f"table_{metric_name}_mean_only.csv"

        ci_tex = f"table_{metric_name}_ci.tex"
        mean_tex = f"table_{metric_name}_mean_only.tex"

        tab_ci.to_csv(ci_csv)
        tab_mean.to_csv(mean_csv)

        tab_ci.to_latex(ci_tex)
        tab_mean.to_latex(mean_tex)

        print(f"Saved: {ci_csv}")
        print(f"Saved: {mean_csv}")
        print(f"Saved: {ci_tex}")
        print(f"Saved: {mean_tex}")






