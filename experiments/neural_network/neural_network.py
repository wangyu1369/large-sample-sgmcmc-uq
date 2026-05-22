import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 0. Global settings
# ============================================================
BASE_SEED = 123
DEVICE = torch.device("cpu")
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 1. Dataset
# ============================================================
def load_dataset():
    ds = load_diabetes()
    X = ds.data
    y = ds.target.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y).reshape(-1)

    X = torch.tensor(X, dtype=DTYPE, device=DEVICE)
    y = torch.tensor(y, dtype=DTYPE, device=DEVICE)
    return X, y


# ============================================================
# 2. Two-hidden-layer neural network: hidden dims (2, 3)
# ============================================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(2, 3)):
        super().__init__()
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


# ============================================================
# 3. Flatten / unflatten parameters
# ============================================================
def flatten_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def get_param_shapes(model):
    return [p.shape for p in model.parameters()]


def get_param_numels(model):
    return [p.numel() for p in model.parameters()]


def unflatten_to_param_list(theta_flat, shapes, numels):
    params = []
    offset = 0
    for shape, n in zip(shapes, numels):
        params.append(theta_flat[offset:offset + n].view(shape))
        offset += n
    return params


def set_model_params_(model, theta_flat, shapes, numels):
    with torch.no_grad():
        offset = 0
        for p, shape, n in zip(model.parameters(), shapes, numels):
            p.copy_(theta_flat[offset:offset + n].view(shape))
            offset += n


# ============================================================
# 4. Functional forward / loss from flat params
# ============================================================
def model_forward_from_flat(theta_flat, x, input_dim, hidden_dims, shapes, numels):
    W1, b1, W2, b2, W3, b3 = unflatten_to_param_list(theta_flat, shapes, numels)

    h1 = torch.tanh(x @ W1.T + b1)
    h2 = torch.tanh(h1 @ W2.T + b2)
    out = h2 @ W3.T + b3

    return out.squeeze(-1)


def loss_per_sample(theta_flat, x_i, y_i, input_dim, hidden_dims, shapes, numels):
    pred = model_forward_from_flat(
        theta_flat,
        x_i.unsqueeze(0),
        input_dim,
        hidden_dims,
        shapes,
        numels,
    )[0]
    return 0.5 * (pred - y_i) ** 2


def batch_loss(theta_flat, x_batch, y_batch, input_dim, hidden_dims, shapes, numels):
    preds = model_forward_from_flat(
        theta_flat,
        x_batch,
        input_dim,
        hidden_dims,
        shapes,
        numels,
    )
    return 0.5 * torch.mean((preds - y_batch) ** 2)


# ============================================================
# 5. Train to a center
# ============================================================
def fit_to_convergence(model, X, y, lr=1e-2, batch_size=32, epochs=300):
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    loss_path = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        count = 0

        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = 0.5 * torch.mean((out - yb) ** 2)
            loss.backward()
            opt.step()

            running += loss.item() * xb.shape[0]
            count += xb.shape[0]

        loss_path.append(running / count)

    return loss_path


# ============================================================
# 6. Fixed-lr SGD tail sampling
# ============================================================
def state_dict_to_numpy(model):
    return flatten_params(model).detach().cpu().numpy()


def run_fixed_lr_sgd_tail(
    model,
    X,
    y,
    lr,
    batch_size,
    burnin_epochs,
    tail_epochs,
):
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    tail_samples = []
    full_loss_path = []

    total_epochs = burnin_epochs + tail_epochs

    for epoch in range(total_epochs):
        model.train()
        running = 0.0
        count = 0

        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = 0.5 * torch.mean((out - yb) ** 2)
            loss.backward()
            opt.step()

            running += loss.item() * xb.shape[0]
            count += xb.shape[0]

            if epoch >= burnin_epochs:
                tail_samples.append(state_dict_to_numpy(model))

        full_loss_path.append(running / count)

    tail_samples = np.array(tail_samples)
    return tail_samples, full_loss_path


# ============================================================
# 7. Empirical mean / covariance
# ============================================================
def empirical_mean_and_cov(samples):
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples.T)
    return mean, cov


# ============================================================
# 8. Curvature objects
# ============================================================
def compute_gradient_and_hessian_single_data_point(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([g.reshape(-1) for g in grads])

    hessian_rows = []

    for g_i in grads_flat:
        second_order = torch.autograd.grad(
            g_i,
            model.parameters(),
            retain_graph=True,
        )
        row = torch.cat([g.reshape(-1) for g in second_order]).detach()
        hessian_rows.append(row)

    H = torch.stack(hessian_rows)
    return grads_flat.detach(), H


def compute_curvature_objects(model, X, y, max_points=None, seed=123):
    n = X.shape[0]
    idx = np.arange(n)

    if max_points is not None and max_points < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)

    J_accum = None
    V_accum = None
    Jn_list = []

    for i in idx:
        loss_i = 0.5 * (model(X[i:i + 1]).squeeze() - y[i]) ** 2
        g_i, H_i = compute_gradient_and_hessian_single_data_point(loss_i, model)

        g_np = g_i.cpu().numpy()
        H_np = H_i.cpu().numpy()

        if J_accum is None:
            d = len(g_np)
            J_accum = np.zeros((d, d))
            V_accum = np.zeros((d, d))

        J_accum += H_np
        V_accum += np.outer(g_np, g_np)
        Jn_list.append(H_np)

    J = J_accum / len(idx)
    V = V_accum / len(idx)

    return J, V, Jn_list


# ============================================================
# 9. Stationary covariance equations
# ============================================================
def stationary_cov_noise_exact(batch_size, cov, J, V, Jn_list):
    C = np.zeros_like(J)

    for H_i in Jn_list:
        C += H_i @ cov @ H_i

    C /= len(Jn_list)
    C -= J @ cov @ J
    C += V
    C /= batch_size

    return C


def stationary_cov_noise_const(batch_size, J):
    return J / batch_size


def continuous_time_equation(cov_flat, lr, batch_size, J):
    d = J.shape[0]
    cov = cov_flat.reshape(d, d)
    cov = 0.5 * (cov + cov.T)

    Lambda = lr * np.eye(d)
    C = stationary_cov_noise_const(batch_size, J)

    eq = Lambda @ J @ cov + cov @ J @ Lambda - Lambda @ C @ Lambda

    return eq.flatten()


def discrete_const_equation(cov_flat, lr, batch_size, J):
    d = J.shape[0]
    cov = cov_flat.reshape(d, d)
    cov = 0.5 * (cov + cov.T)

    Lambda = lr * np.eye(d)
    C = stationary_cov_noise_const(batch_size, J)

    eq = (
        Lambda @ J @ cov
        + cov @ J @ Lambda
        - Lambda @ C @ Lambda
        - Lambda @ J @ cov @ J @ Lambda
    )

    return eq.flatten()


def discrete_exact_equation(cov_flat, lr, batch_size, J, V, Jn_list):
    d = J.shape[0]
    cov = cov_flat.reshape(d, d)
    cov = 0.5 * (cov + cov.T)

    Lambda = lr * np.eye(d)
    C = stationary_cov_noise_exact(batch_size, cov, J, V, Jn_list)

    eq = (
        Lambda @ J @ cov
        + cov @ J @ Lambda
        - Lambda @ C @ Lambda
        - Lambda @ J @ cov @ J @ Lambda
    )

    return eq.flatten()


def solve_covariance_equation(fun, d, initial_guess, args):
    sol = root(fun, x0=initial_guess, args=args, method="hybr")
    cov = sol.x.reshape(d, d)
    cov = 0.5 * (cov + cov.T)

    return cov, sol


# ============================================================
# 10. Diagnostics
# ============================================================
def relative_frobenius(emp_cov, theory_cov):
    return np.linalg.norm(emp_cov - theory_cov, ord="fro") / (
        np.linalg.norm(theory_cov, ord="fro") + 1e-12
    )


def project_to_psd(A, eps=1e-10):
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)

    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ============================================================
# 11. One replicate
# ============================================================
def run_one_replicate(
    rep_seed,
    hidden_dims=(2, 3),
    init_train_lr=1e-2,
    init_train_batch_size=32,
    init_train_epochs=300,
    lr_list=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
    retrain_batch_size=32,
    burnin_epochs=50,
    tail_epochs=100,
    curvature_subset=None,
    make_plot=False,
):
    set_seed(rep_seed)

    X, y = load_dataset()

    model = SimpleNN(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
    ).to(DEVICE, dtype=DTYPE)

    shapes = get_param_shapes(model)
    numels = get_param_numels(model)

    loss_path = fit_to_convergence(
        model,
        X,
        y,
        lr=init_train_lr,
        batch_size=init_train_batch_size,
        epochs=init_train_epochs,
    )

    if make_plot:
        plt.figure(figsize=(5, 4))
        plt.plot(loss_path)
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title(f"Initial training loss, seed={rep_seed}")
        plt.tight_layout()
        plt.show()

    trained_params = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }

    result = {}
    param_dim = int(sum(p.numel() for p in model.parameters()))
    result["param_dim"] = param_dim

    for lr in lr_list:
        model.load_state_dict(trained_params)

        tail_samples, retrain_loss_path = run_fixed_lr_sgd_tail(
            model,
            X,
            y,
            lr=lr,
            batch_size=retrain_batch_size,
            burnin_epochs=burnin_epochs,
            tail_epochs=tail_epochs,
        )

        theta_hat, emp_cov = empirical_mean_and_cov(tail_samples)

        set_model_params_(
            model,
            torch.tensor(theta_hat, dtype=DTYPE, device=DEVICE),
            shapes,
            numels,
        )

        J, V, Jn_list = compute_curvature_objects(
            model,
            X,
            y,
            max_points=curvature_subset,
            seed=rep_seed,
        )

        d = J.shape[0]
        initial_guess = np.eye(d).flatten()

        # Continuous-time
        ct_cov, ct_sol = solve_covariance_equation(
            continuous_time_equation,
            d,
            initial_guess,
            (lr, retrain_batch_size, J),
        )
        ct_cov = project_to_psd(ct_cov)

        # Discrete-time + constant noise
        dq_const_cov, dq_const_sol = solve_covariance_equation(
            discrete_const_equation,
            d,
            initial_guess,
            (lr, retrain_batch_size, J),
        )
        dq_const_cov = project_to_psd(dq_const_cov)

        # Discrete-time + exact noise
        dq_exact_cov, dq_exact_sol = solve_covariance_equation(
            discrete_exact_equation,
            d,
            initial_guess,
            (lr, retrain_batch_size, J, V, Jn_list),
        )
        dq_exact_cov = project_to_psd(dq_exact_cov)

        result[lr] = {
            "CT": relative_frobenius(emp_cov, ct_cov),
            "DQ+const": relative_frobenius(emp_cov, dq_const_cov),
            "DQ+exact": relative_frobenius(emp_cov, dq_exact_cov),
            "CT_success": bool(ct_sol.success),
            "DQ+const_success": bool(dq_const_sol.success),
            "DQ+exact_success": bool(dq_exact_sol.success),
        }

    return result


# ============================================================
# 12. Repeated experiment
# ============================================================
def run_repeated_experiment(
    n_reps=20,
    hidden_dims=(2, 3),
    init_train_lr=1e-2,
    init_train_batch_size=32,
    init_train_epochs=300,
    lr_list=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5),
    retrain_batch_size=32,
    burnin_epochs=50,
    tail_epochs=100,
    curvature_subset=None,
):
    all_results = []

    for rep in range(n_reps):
        rep_seed = BASE_SEED + rep

        print("\n" + "=" * 80)
        print(f"REPLICATE {rep + 1}/{n_reps} | seed={rep_seed}")
        print("=" * 80)

        rep_result = run_one_replicate(
            rep_seed=rep_seed,
            hidden_dims=hidden_dims,
            init_train_lr=init_train_lr,
            init_train_batch_size=init_train_batch_size,
            init_train_epochs=init_train_epochs,
            lr_list=lr_list,
            retrain_batch_size=retrain_batch_size,
            burnin_epochs=burnin_epochs,
            tail_epochs=tail_epochs,
            curvature_subset=curvature_subset,
            make_plot=(rep == 0),
        )

        all_results.append(rep_result)

        print(f"parameter dimension = {rep_result['param_dim']}")

        for lr in lr_list:
            print(
                f"lr={lr:.4f} | "
                f"CT={rep_result[lr]['CT']:.6e} | "
                f"DQ+const={rep_result[lr]['DQ+const']:.6e} | "
                f"DQ+exact={rep_result[lr]['DQ+exact']:.6e}"
            )

    return all_results


# ============================================================
# 13. Aggregate raw results
# ============================================================
def aggregate_raw_results(all_results, lr_list):
    methods = ["CT", "DQ+const", "DQ+exact"]
    raw_rows = []

    for rep_idx, rep_result in enumerate(all_results):
        for lr in lr_list:
            row = {
                "replicate": rep_idx + 1,
                "lr": lr,
            }

            for method in methods:
                row[method] = rep_result[lr][method]

            raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    return raw_df


# ============================================================
# 14. Mean CI summary and styled plot
# ============================================================
def summarize_with_mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = np.mean(values)

    if n <= 1:
        return {
            "mean": mean,
            "lower": mean,
            "upper": mean,
        }

    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    z = 1.96

    return {
        "mean": mean,
        "lower": mean - z * se,
        "upper": mean + z * se,
    }


def build_summary_from_raw(raw_df):
    methods = ["CT", "DQ+const", "DQ+exact"]
    lr_list = sorted(raw_df["lr"].unique())

    summary_rows = []

    for lr in lr_list:
        row = {"lr": lr}
        sub = raw_df.loc[raw_df["lr"] == lr]

        for method in methods:
            stats = summarize_with_mean_ci(sub[method].values)

            row[f"{method}_mean"] = stats["mean"]
            row[f"{method}_lower"] = stats["lower"]
            row[f"{method}_upper"] = stats["upper"]
            row[f"{method}_display"] = (
                f"{stats['mean']:.6f} "
                f"[{stats['lower']:.6f}, {stats['upper']:.6f}]"
            )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def plot_from_raw_df(raw_df, save_path="neural_network.pdf"):
    summary_df = build_summary_from_raw(raw_df)

    plt.figure(figsize=(7.2, 4.5))

    method_styles = {
        "CT": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "label": "CT",
        },
        "DQ+const": {
            "color": "#ff7f0e",
            "marker": "s",
            "linestyle": "--",
            "label": "DQ+const",
        },
        "DQ+exact": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyle": "-.",
            "label": "DQ+exact (this paper)",
        },
    }

    for method, style in method_styles.items():
        x = summary_df["lr"].to_numpy()
        mean = summary_df[f"{method}_mean"].to_numpy()
        lower = summary_df[f"{method}_lower"].to_numpy()
        upper = summary_df[f"{method}_upper"].to_numpy()

        plt.plot(
            x,
            mean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5,
            label=style["label"],
        )

        plt.fill_between(
            x,
            lower,
            upper,
            color=style["color"],
            alpha=0.18,
        )

    plt.xscale("log")

    plt.xlabel(r"Learning rate $\lambda$", fontsize=14)
    plt.ylabel(
        r"Covariance error: $\|\Sigma_{\psi}-\Sigma_{\theta}\|_{F} / \|\Sigma_{\theta}\|_{F}$",
        fontsize=13,
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    plt.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

    return summary_df


def print_summary_table(summary_df, n_reps):
    print("\n" + "=" * 100)
    print(f"SUMMARY: relative Frobenius error with 95% CI for the mean across {n_reps} replicates")
    print("=" * 100)

    display_cols = [
        "lr",
        "CT_display",
        "DQ+const_display",
        "DQ+exact_display",
    ]

    print(summary_df[display_cols].to_string(index=False))


# ============================================================
# 15. Main
# ============================================================
if __name__ == "__main__":
    # ----- config -----
    hidden_dims = (2, 3)

    init_train_lr = 1e-2
    init_train_batch_size = 32
    init_train_epochs = 300

    lr_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    retrain_batch_size = 32
    burnin_epochs = 50
    tail_epochs = 100

    curvature_subset = None
    # For faster debugging, you can use:
    # curvature_subset = 200

    n_reps = 30

    # ----- run repeated experiment -----
    all_results = run_repeated_experiment(
        n_reps=n_reps,
        hidden_dims=hidden_dims,
        init_train_lr=init_train_lr,
        init_train_batch_size=init_train_batch_size,
        init_train_epochs=init_train_epochs,
        lr_list=lr_list,
        retrain_batch_size=retrain_batch_size,
        burnin_epochs=burnin_epochs,
        tail_epochs=tail_epochs,
        curvature_subset=curvature_subset,
    )

    # ----- aggregate raw results -----
    raw_df = aggregate_raw_results(
        all_results=all_results,
        lr_list=lr_list,
    )

    # ----- build summary and plot directly from raw_df -----
    summary_df = plot_from_raw_df(
        raw_df,
        save_path="neural_network.pdf",
    )

    # ----- save results -----
    raw_df.to_csv(
        "raw_results_diabetes_nn_2layer_20reps.csv",
        index=False,
    )

    summary_df.to_csv(
        "summary_results_diabetes_nn_2layer_20reps.csv",
        index=False,
    )

    # ----- print summary -----
    print_summary_table(summary_df, n_reps)

    print("\nSaved:")
    print("  raw_results_diabetes_nn_2layer_20reps.csv")
    print("  summary_results_diabetes_nn_2layer_20reps.csv")
    print("  neural_network.pdf")