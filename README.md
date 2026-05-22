**Accurate Large-sample Uncertainty Quantification using Stochastic Gradient Markov Chain Monte Carlo**
## Experiments

This repository contains code for reproducing the main experiments in the paper. The experiments are organized as follows.

### 1. Motivating Example

The motivating example illustrates the limitation of continuous-time/SDE-based approximations for fixed-step-size SGD/SGLD. In particular, it compares the stationary covariance predicted by existing continuous-time theory with the empirical stationary covariance observed from discrete-time SGD iterates.

**Code location:**

```text
experiments/motivating_example/
