# Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo

## Experiments

This repository contains code for reproducing the main experiments in the paper. The experiments are organized as follows.

---

### 1. Motivating Example

The motivating example, corresponding to Figure 1 in the paper, illustrates the limitation of continuous-time/SDE-based approximations for SGD/SGLD when the batch size or learning rate is not sufficiently small. In this regime, continuous-time approximations can provide inaccurate predictions of the stationary covariance, motivating the need for a discrete-time characterization.

**Code location:**

```text
experiments/motivating_example/motivating_plot.py
```

---

### 2. Robust Linear Regression

The robust linear regression experiments compare tuning instructions suggested by different theories when the goal is to target the sandwich covariance

$$
\mathcal{S}_{\star}
=
\mathcal{J}_{\star}^{-1}
\mathcal{I}_{\star}
\mathcal{J}_{\star}^{-1}.
$$

In particular, we evaluate whether different theory-based choices of the tuning parameters lead SGD/SGLD to a stationary distribution whose covariance matches the desired sandwich covariance.

#### 2(a). Simulation

The simulation experiment uses synthetic misspecified data with outliers. This controlled setting allows us to compare different theory-based tuning rules and evaluate how accurately they recover the target sandwich covariance.

**Code location:**

```text
experiments/robust_linear_regression/simulation/robust_linear_regression_simulation.py
```

#### 2(b). Real-world Dataset

The real-world robust linear regression experiment evaluates the same uncertainty quantification procedure on the Boston Housing dataset. We compare the stationary covariance induced by different tuning rules with the target sandwich covariance.

**Code location:**

```text
experiments/robust_linear_regression/real_data/robust_regression_boston.py
```

---

### 3. Poisson Regression

The Poisson regression experiments study uncertainty quantification for generalized linear models with count data. As in the robust linear regression experiments, the goal is to compare tuning instructions suggested by different theories when targeting the sandwich covariance

$$
\mathcal{S}_{\star}
=
\mathcal{J}_{\star}^{-1}
\mathcal{I}_{\star}
\mathcal{J}_{\star}^{-1}.
$$

We compare whether the resulting stationary covariance from SGD/SGLD matches the desired sandwich covariance under both synthetic and real-world settings.

#### 3(a). Simulation

The simulation experiment uses synthetic well-specified Poisson regression data. This setting evaluates whether different theory-based tuning rules recover the target covariance when the model is correctly specified.

**Code location:**

```text
experiments/poisson_regression/simulation/poisson_regression_simulation.py
```

#### 3(b). Real-world Dataset

The real-world Poisson regression experiment evaluates the proposed method on the German Credit dataset. We compare the stationary covariance induced by different tuning rules with the target sandwich covariance.

**Code location:**

```text
experiments/poisson_regression/real_data/poisson_regression_german_data.py
```

---

### 4. Neural Network

The neural-network experiment evaluates stationary covariance approximation beyond the convex setting considered in the main theory. We fit a small neural network and compare the stationary covariance predicted by different theories under a range of fixed learning rates.

The goal is to assess how accurately each theory predicts the empirical stationary covariance of SGD iterates. This experiment illustrates that continuous-time approximations can be accurate at small learning rates but become less reliable as the learning rate increases, while the proposed discrete-time covariance characterization remains informative in a nonconvex model.

**Code location:**

```text
experiments/neural_network/neural_network.py
```

---

### 5. Wasserstein Bound Validation

We empirically validate the Wasserstein bound using Poisson regression in both well-specified and misspecified settings. Specifically, we consider:  
(i) synthetic Poisson data fitted with a correctly specified Poisson model, and  
(ii) synthetic negative binomial data fitted with a misspecified Poisson model.

We then examine whether the observed Wasserstein distance follows the scaling predicted by our theory.

**Code location:**

```text
experiments/wasserstein_bound/run_wasserstein_validation.py
```
experiments/wasserstein_bound/wasserstein_bound.py
```
