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

The real-world robust linear regression experiment evaluates uncertainty quantification on real-world Boston Housing datasets.

**Code location:**

```text
experiments/robust_linear_regression/real_data/
```

### 3. Poisson Regression

The Poisson regression experiments study uncertainty quantification for generalized linear models with count data. We compare continuous-time and discrete-time stationary covariance predictions under both synthetic and real-world settings.

#### 3(a). Simulation

The simulation experiment uses synthetic count data generated from a controlled data-generating process. This allows us to evaluate covariance approximation accuracy under well-specified and/or misspecified Poisson regression settings.

**Code location:**

```text
experiments/poisson_regression/simulation/
```

#### 3(b). Real-world Dataset

The real-world Poisson regression experiment evaluates the proposed method on real-world count-data regression tasks. The goal is to assess whether the discrete-time covariance approximation remains accurate beyond synthetic examples.

**Code location:**

```text
experiments/poisson_regression/real_data/
```

---

### 4. Neural Network

The neural-network experiment evaluates stationary covariance approximation beyond the convex setting considered in the main finite-sample theory. We fit a small neural network and compare the stationary covariance predicted by different theories with the empirical covariance estimated from SGD iterates.

This experiment illustrates that the proposed discrete-time covariance characterization can remain informative for nonconvex models.

**Code location:**

```text
experiments/neural_network/
```

---

### 5. Wasserstein Bound Validation

The Wasserstein bound validation experiment empirically studies the distance between stationary distributions induced by different proxy algorithms. We examine how the Wasserstein distance scales with the step size and minibatch size, and compare the observed trend with the theoretical prediction.

**Code location:**

```text
experiments/wasserstein_bound/
```
