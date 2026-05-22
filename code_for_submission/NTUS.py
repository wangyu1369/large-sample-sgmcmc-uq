import autograd.numpy as np
from autograd import grad, hessian
from autograd.scipy.stats import norm
from scipy.special import logsumexp

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph.op import Op
except ImportError:
    pm = None
    pt = None
    Op = object


# ------------------------------
# Data generator
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

        self.cov_x = (np.linalg.inv(cov_theta) - penalty_para * np.eye(self.p)) / self.N

    def generate_data(self, outlier_frac=0.0, outlier_scale=20.0, outlier_shift=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)

        mean = np.zeros(self.p)
        X = np.random.multivariate_normal(mean, self.cov_x, self.N)

        var_eps = np.array([1.0 + np.sum(X[i] ** 2) for i in range(self.N)])
        mean_y = X.dot(self.true_theta) + self.bias
        Y = np.random.normal(mean_y, np.sqrt(var_eps))

        if outlier_frac > 0:
            n_out = int(outlier_frac * self.N)
            idx = np.random.choice(self.N, size=n_out, replace=False)
            Y[idx] = outlier_shift + np.random.normal(mean_y[idx], outlier_scale * np.sqrt(var_eps[idx]))

        return X, Y


# ============================================================
# Losses
# ============================================================
def f_density(y, x, theta, sigma=1.0):
    mu = np.dot(x, theta)
    return norm.pdf(y, mu, sigma)

def beta_loss(theta, X, Y, beta=1.5, sigma=1.0, n_mc=200, z_samples=None):
    N, _ = X.shape
    if z_samples is None:
        z_samples = np.random.normal(0, sigma * 5, size=n_mc)

    total = 0.0
    for i in range(N):
        fy = f_density(Y[i], X[i], theta, sigma)
        mu_i = np.dot(X[i], theta)
        fz = norm.pdf(z_samples, mu_i, sigma)
        integral_term = np.mean(fz ** beta)
        total += (-fy ** (beta - 1) / (beta - 1) + integral_term / beta)

    return total / N

def beta_loss_per_sample(theta, y, x, beta=1.5, sigma=1.0, n_mc=200, z_samples=None):
    if z_samples is None:
        z_samples = np.random.normal(0, sigma * 5, size=n_mc)
    fy = f_density(y, x, theta, sigma)
    fz = f_density(z_samples, x, theta, sigma)
    integral_term = np.mean(fz ** beta)
    return -fy ** (beta - 1) / (beta - 1) + integral_term / beta

def nll_loss(theta, X, Y, sigma=1.0):
    r = (Y - X @ theta) / sigma
    return np.mean(0.5 * r**2 + np.log(sigma) + 0.5 * np.log(2 * np.pi))

def nll_loss_per_sample(theta, y, x, sigma=1.0):
    r = (y - np.dot(x, theta)) / sigma
    return 0.5 * r**2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)


# ============================================================
# Optimizer
# ============================================================
def rmsprop(loss_fn, theta_init, X, Y, lr=0.05, decay=0.9, eps=1e-8, n_iter=200, print_every=None, **loss_kwargs):
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
# J and V
# ============================================================
def empirical_fisher_from_per_sample_grad(theta_hat, X, Y, per_sample_loss_fn, per_sample_kwargs):
    N, D = X.shape
    g_fn = grad(lambda th, yi, xi: per_sample_loss_fn(th, yi, xi, **per_sample_kwargs))
    V = np.zeros((D, D))
    for i in range(N):
        g = g_fn(theta_hat, Y[i], X[i]).reshape(-1, 1)
        V += g @ g.T
    return V / N

def empirical_hessian_mean(theta_hat, X, Y, per_sample_loss_fn, per_sample_kwargs):
    N, D = X.shape
    h_fn = hessian(lambda th, yi, xi: per_sample_loss_fn(th, yi, xi, **per_sample_kwargs))
    Hs = np.zeros((N, D, D))
    for i in range(N):
        Hs[i] = h_fn(theta_hat, Y[i], X[i])
    J = np.mean(Hs, axis=0)
    return Hs, J


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
    thetas = np.asarray(theta_samples[idx])

    mu = np.asarray(X_test) @ thetas.T
    logp = norm.logpdf(np.asarray(Y_test)[:, None], loc=mu, scale=sigma)
    log_pred = logsumexp(logp, axis=1) - np.log(logp.shape[1])
    return -np.mean(log_pred)

def metrics_for_reference(samples, theta_target, cov_target, X_test, Y_test, sigma_eval=1.0):
    mean = samples.mean(axis=0)
    emp_cov = np.cov(samples, rowvar=False, bias=True)

    pe = np.linalg.norm(mean - theta_target) / (np.linalg.norm(theta_target) + 1e-12)
    cov_norm = np.linalg.norm(cov_target, ord="fro") + 1e-12
    frob = np.linalg.norm(emp_cov - cov_target, ord="fro") / cov_norm
    nllv = predictive_nll_from_samples(samples, X_test, Y_test, sigma=sigma_eval, max_samples=1000, seed=0)
    ks = ks_to_uniform(quantile_calibration(samples, theta_target))
    return pe, frob, nllv, ks


# ============================================================
# NUTS helpers
# ============================================================
def log_gaussian_prior(theta, tau2=25.0):
    d = theta.shape[0]
    return -0.5 * np.sum(theta**2) / tau2 - 0.5 * d * np.log(2 * np.pi * tau2)

def log_posterior_nll(theta, X, Y, sigma=1.0, tau2=25.0):
    N = X.shape[0]
    return -N * nll_loss(theta, X, Y, sigma=sigma) + log_gaussian_prior(theta, tau2=tau2)

def log_posterior_beta(theta, X, Y, beta=1.5, sigma=1.0, n_mc=200, z_samples=None, tau2=25.0):
    N = X.shape[0]
    return -N * beta_loss(theta, X, Y, beta=beta, sigma=sigma, n_mc=n_mc, z_samples=z_samples) \
           + log_gaussian_prior(theta, tau2=tau2)

class LogPGradOp(Op):
    itypes = [pt.dvector] if pt is not None else []
    otypes = [pt.dvector] if pt is not None else []

    def __init__(self, grad_fn):
        self.grad_fn = grad_fn

    def perform(self, node, inputs, outputs):
        theta_val = inputs[0]
        outputs[0][0] = np.array(self.grad_fn(theta_val), dtype=np.float64)

class LogPOp(Op):
    itypes = [pt.dvector] if pt is not None else []
    otypes = [pt.dscalar] if pt is not None else []

    def __init__(self, logp_fn, grad_fn):
        self.logp_fn = logp_fn
        self.logpgrad = LogPGradOp(grad_fn)

    def perform(self, node, inputs, outputs):
        theta_val = inputs[0]
        outputs[0][0] = np.array(self.logp_fn(theta_val), dtype=np.float64)

    def grad(self, inputs, g_outputs):
        theta = inputs[0]
        return [g_outputs[0] * self.logpgrad(theta)]

def nuts_samples_from_logp(logp_fn, dim, n_samples=300, n_tune=300, n_chains=1, seed=0, initval=None):
    if pm is None or pt is None:
        raise ImportError("PyMC and PyTensor are required for NUTS. Install with: pip install pymc")

    grad_fn = grad(logp_fn)
    logp_op = LogPOp(logp_fn, grad_fn)

    if initval is None:
        initval = np.zeros(dim)

    with pm.Model() as model:
        theta = pm.Flat("theta", shape=dim, initval=initval)
        pm.Potential("target_logp", logp_op(theta))

        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            cores=1,
            target_accept=0.9,
            init="adapt_diag",
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    arr = trace.posterior["theta"].stack(sample=("chain", "draw")).values.T
    return np.asarray(arr)


# ============================================================
# One replication: NUTS only, no batch-size output
# ============================================================
def run_one_rep_nuts_reference(rep_seed: int,
                               dimension=20,
                               sample_size=1000,
                               beta=1.5,
                               sigma_working=1.0,
                               n_mc=20,
                               outlier_frac=0.1,
                               outlier_scale=5.0,
                               outlier_shift=10.0,
                               tau2_prior=25.0,
                               nuts_draws=300,
                               nuts_tune=300,
                               nuts_chains=1):
    if pd is None:
        raise ImportError("Please install pandas.")

    np.random.seed(rep_seed)

    # ---- data
    model = NonlinearDependentNoise(N=sample_size, p=dimension, rho=0.0, bias=1.0, seed=rep_seed + 123)
    X, Y = model.generate_data(outlier_frac=outlier_frac,
                               outlier_scale=outlier_scale,
                               outlier_shift=outlier_shift,
                               seed=rep_seed + 456)
    theta_true = np.asarray(model.true_theta)

    # ---- test data
    oldN = model.N
    model.N = 3000
    X_test, Y_test = model.generate_data(outlier_frac=outlier_frac,
                                         outlier_scale=outlier_scale,
                                         outlier_shift=outlier_shift,
                                         seed=rep_seed + 789)
    model.N = oldN

    # ---- deterministic z for beta-loss
    rng_beta = np.random.default_rng(rep_seed + 999)
    z_fixed = rng_beta.normal(0.0, sigma_working * 5.0, size=n_mc)

    # ---- beta fit
    theta_init = np.zeros(dimension)
    theta_hat_beta = rmsprop(
        beta_loss, theta_init, X, Y,
        lr=0.1, n_iter=120, print_every=None,
        beta=beta, sigma=sigma_working, n_mc=n_mc, z_samples=z_fixed
    )

    beta_kwargs = dict(beta=beta, sigma=sigma_working, n_mc=n_mc, z_samples=z_fixed)
    V_beta = empirical_fisher_from_per_sample_grad(theta_hat_beta, X, Y, beta_loss_per_sample, beta_kwargs)
    _, J_beta = empirical_hessian_mean(theta_hat_beta, X, Y, beta_loss_per_sample, beta_kwargs)
    Jinv_beta = np.linalg.inv(J_beta)
    sandwich_beta = Jinv_beta @ V_beta @ Jinv_beta
    scaled_sandwich_beta = sandwich_beta / sample_size

    # ---- nll fit
    XtX = np.asarray(X.T @ X)
    XtY = np.asarray(X.T @ Y)
    theta_hat_nll = np.linalg.solve(XtX, XtY)

    nll_kwargs = dict(sigma=sigma_working)
    V_nll = empirical_fisher_from_per_sample_grad(theta_hat_nll, X, Y, nll_loss_per_sample, nll_kwargs)
    _, J_nll = empirical_hessian_mean(theta_hat_nll, X, Y, nll_loss_per_sample, nll_kwargs)
    Jinv_nll = np.linalg.inv(J_nll)
    sandwich_nll = Jinv_nll @ V_nll @ Jinv_nll
    scaled_sandwich_nll = sandwich_nll / sample_size

    # ---- NUTS references
    nuts_nll_samples = nuts_samples_from_logp(
        logp_fn=lambda th: log_posterior_nll(th, X, Y, sigma=sigma_working, tau2=tau2_prior),
        dim=dimension,
        n_samples=nuts_draws,
        n_tune=nuts_tune,
        n_chains=nuts_chains,
        seed=rep_seed + 333,
        initval=theta_hat_nll,
    )

    nuts_beta_samples = nuts_samples_from_logp(
        logp_fn=lambda th: log_posterior_beta(
            th, X, Y, beta=beta, sigma=sigma_working, n_mc=n_mc,
            z_samples=z_fixed, tau2=tau2_prior
        ),
        dim=dimension,
        n_samples=nuts_draws,
        n_tune=nuts_tune,
        n_chains=nuts_chains,
        seed=rep_seed + 444,
        initval=theta_hat_beta,
    )

    pe_nuts_nll, frob_nuts_nll, nll_nuts_nll, ks_nuts_nll = metrics_for_reference(
        nuts_nll_samples, theta_true, scaled_sandwich_nll, X_test, Y_test, sigma_eval=sigma_working
    )
    pe_nuts_beta, frob_nuts_beta, nll_nuts_beta, ks_nuts_beta = metrics_for_reference(
        nuts_beta_samples, theta_true, scaled_sandwich_beta, X_test, Y_test, sigma_eval=sigma_working
    )

    rows = [
        {
            "rep": int(rep_seed),
            "method": "NUTS (log loss)",
            "param_err": float(pe_nuts_nll),
            "frob_cov": float(frob_nuts_nll),
            "pred_nll": float(nll_nuts_nll),
            "calib_ks": float(ks_nuts_nll),
        },
        {
            "rep": int(rep_seed),
            "method": r"NUTS ($\beta = 1.5$)",
            "param_err": float(pe_nuts_beta),
            "frob_cov": float(frob_nuts_beta),
            "pred_nll": float(nll_nuts_beta),
            "calib_ks": float(ks_nuts_beta),
        },
    ]

    return pd.DataFrame(rows), nuts_nll_samples, nuts_beta_samples


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if pd is None:
        raise SystemExit("This script needs pandas. Install via: pip install pandas")
    if pm is None:
        raise SystemExit("This script needs PyMC. Install via: pip install pymc")

    DIMENSION = 20
    SAMPLE_SIZE = 1000

    BETA = 1.5
    SIGMA_WORKING = 1.0
    N_MC = 20

    OUTLIER_FRAC = 0.1
    OUTLIER_SCALE = 5.0
    OUTLIER_SHIFT = 10.0

    TAU2_PRIOR = 25.0
    NUTS_DRAWS = 300
    NUTS_TUNE = 300
    NUTS_CHAINS = 1

    R = 1
    REP_SEEDS = [100]

    dfs = []
    all_nll_samples = []
    all_beta_samples = []

    for s in REP_SEEDS:
        print(f"Running NUTS reference replication {s}")
        df_rep, nuts_nll_samples, nuts_beta_samples = run_one_rep_nuts_reference(
            rep_seed=s,
            dimension=DIMENSION,
            sample_size=SAMPLE_SIZE,
            beta=BETA,
            sigma_working=SIGMA_WORKING,
            n_mc=N_MC,
            outlier_frac=OUTLIER_FRAC,
            outlier_scale=OUTLIER_SCALE,
            outlier_shift=OUTLIER_SHIFT,
            tau2_prior=TAU2_PRIOR,
            nuts_draws=NUTS_DRAWS,
            nuts_tune=NUTS_TUNE,
            nuts_chains=NUTS_CHAINS,
        )
        dfs.append(df_rep)
        all_nll_samples.append(nuts_nll_samples)
        all_beta_samples.append(nuts_beta_samples)

    df_nuts = pd.concat(dfs, ignore_index=True)
    df_nuts.to_csv("results_nuts_reference.csv", index=False)
    print("Saved: results_nuts_reference.csv")

    # Save samples too
    np.save("nuts_log_loss_samples.npy", all_nll_samples[0])
    np.save("nuts_beta_loss_samples.npy", all_beta_samples[0])
    print("Saved: nuts_log_loss_samples.npy")
    print("Saved: nuts_beta_loss_samples.npy")

    print("\nPreview:")
    print(df_nuts.head())
    
