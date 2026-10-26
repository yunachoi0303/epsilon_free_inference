# Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation

Code for reproducing the experiments in the paper:

> G. Papamakarios and I. Murray, _Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation_, NeurIPS 2016.
> [[arXiv]](https://arxiv.org/abs/1605.06376) [[bibtex]](https://gpapamak.github.io/bibtex/epsilon_free_inference.bib)


## This folder contains

#### **`demos`**

Folder containing four subfolders, one for each demo in the paper.

* `mixture_of_gaussians_demo`
  - `mog_main.py` --- sets up the model
  - `mog_abc.py` --- runs ABC methods
  - `mog_mdn.py` --- runs MDN methods
  - `mog_res.py` --- collects and plots results

Call the module with `python -m demos.mixture_of_gaussians_demo.mog_main` in the terminal.

* `bayesian_linear_regression_demo`
  - `blr_main.py` --- sets up the model
  - `blr_abc.py` --- runs ABC methods
  - `blr_mdn.py` --- runs MDN methods
  - `blr_res.py` --- collects and plots results

* `lotka_volterra_demo`
  - `lv_main.py` --- sets up the model
  - `lv_abc.py` --- runs ABC methods
  - `lv_mdn.py` --- runs MDN methods
  - `lv_res.py` --- collects and plots results

* `mg1_queue_demo`
  - `mg1_main.py` --- sets up the model
  - `mg1_abc.py` --- runs ABC methods
  - `mg1_mdn.py` --- runs MDN methods
  - `mg1_res.py` --- collects and plots results

#### **`util`**

Folder with utility classes and functions.

* `pdf.py`
    Gaussians and mixtures of Gaussians

* `NeuralNet.py`
  neural nets with and without SVI

* `mdn.py`
  MDNs with and without SVI

* `DataStream.py`
  provides data minibatches for training

* `LossFunction.py`
  loss functions for training

* `StepStrategy.py`
  optimization algorithms, including Adam

* `Trainer.py`
  trains a neural net or MDN, SVI or not

* `MarkovJumpProcess.py`
  Markov jump processes, including Lotka--Volterra
  
* `helper.py`
    various helper functions


#### **`MoG`** 
param a: mixing coefficients
This represent the relative weight or probability assigned to each individual probability distribution (such as a Gaussian component), determining how much each sub-population contributes to the overall dataset.

param ms: means


param Ps: precisions
A performance metric that measures the proportion of positive predictions made by a model that are actually correct. 
Precision = TP / (TP + FP)

param Us: precision factors such that U'U = P
'U' is the upper triangular matrix such that U'*U is the precision

param Ss: covariances


param xs: list of gaussian variables
Two defining parameters are mean (μ) and variance (σ^2).
The emperical rule for any Gaussian distribution 
* 68% of the data falls within μ±1σ
* 95% falls within μ±2σ
* 99.7% falls within μ±3σ


## **Files**

`mdn_svi_proposal_hiddens_20_tanh_sims_1k`
| Position | Name | Description |
| :---     | :--- | :---       |
| 1        | `net`| The trained MDN_SVI network, i.e. its learnt variational weight means and log-stds. It started as the network loaded from `mdn_svi_proposal_prior_<n_bootstrap_iter-1>.pkl`, was expanded to 2 mixture components by `replicate_gaussian_mdn(net, 2)`, and was then trained on the 1000 simulations. |
| 2 | `approx_posterior` | A MoG object: the network’s posterior at `x_obs` with negligible components pruned (threshold `1e-3`), divided by `prior_proposal`. |


`mdn_svi_proposal_prior_<n_bootstrap_iter>.pkl`
| Position | Name | Description |
| :---     | :--- | :---       |
| 1        | `net` | The MDN-SVI after this iteration’s training (1 input, one hidden layer of 20 tanh units, 1 output, 1 component). |
| 2        | `approx_posterior` | A MoG: the network’s posterior at `x_obs`, pruned at `1e-3`. At `iter == 0` it is used as is. At later iterations it is `mdn_mog / prior_proposal`, the SNPE-A correction for having trained on samples from the proposal. |
| 3        | `prior_proposal` | `approx_posterior.project_to_gaussian()`: a single Gaussian with the same mean and covariance as `approx_posterior`. |
| 4        | `dist` | The 200 distances `calc_dist(xs[i], x_obs)` between each simulated data point and the observed one, from this iteration’s simulations. |

