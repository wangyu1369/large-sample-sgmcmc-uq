**Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo**
## Experiments

This repository contains code for reproducing the main experiments in the paper. The experiments are organized as follows.

### 1. Motivating Example

The motivating example (Figure 1) illustrates the limitation of continuous-time/SDE-based approximations for SGD/SGLD with large batch size (or learning rate).

**Code location:**

```text
experiments/motivating_example/
```

### 2. Robust Linear Regression

The robust linear regression experiments evaluate uncertainty quantification under model misspecification. 

#### 2(a). Simulation

The simulation experiment uses synthetic misspecified data with outliers.
**Code location:**

```text
experiments/robust_linear_regression/simulation/
```

#### 2(b). Real-world Dataset

The real-world robust linear regression experiment evaluates uncertainty quantification on real-world Boston Housing dataset.

**Code location:**

```text
experiments/robust_linear_regression/real_data/
```

### 3. Poisson Regression

The Poisson regression experiments study uncertainty quantification for generalized linear models with count data.

#### 3(a). Simulation

The simulation experiment uses synthetic well-specified data for Poisson regression.

**Code location:**

```text
experiments/poisson_regression/simulation/
```

#### 3(b). Real-world Dataset

The real-world Poisson regression experiment evaluates the proposed method on real-world German Credit dataset.

**Code location:**

```text
experiments/poisson_regression/real_data/
```

---

### 4. Neural Network

The neural-network experiment evaluates stationary covariance approximation beyond the convex setting. We fit a small neural network and compare the stationary covariance predicted by different theories with the empirical covariance estimated from SGD iterates.

This experiment illustrates that the proposed discrete-time covariance characterization can remain informative for nonconvex models.

**Code location:**

```text
experiments/neural_network/
```

---

### 5. Wasserstein Bound Validation

We empirically validate the Wasserstein bound using Poisson regression in both well-specified and misspecified settings. We consider (i) synthetic Poisson data fitted with a correctly specified Poisson model and (ii) synthetic negative binomial data fitted with a misspecified Poisson model. We then examine whether the observed Wasserstein distance follows the scaling predicted by our theory.

**Code location:**

```text
experiments/wasserstein_bound/
