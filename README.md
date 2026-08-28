# TAVIE-SSG: A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super-Gaussian Likelihoods

<p align="center">
  <img src="assets/TAVIE-SSG_animated_logo.gif" alt="TAVIE_logo" width="650"/>
</p>

This repository holds the source code and implementation of **TAVIE-SSG** proposed in *A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super‑Gaussian Likelihoods*.

---


## Overview

An overview of **T**angent **A**pproximation based **V**ariational **I**nferenc**E** Framework for **S**trongly **S**uper-**G**aussian Likelihoods (**TAVIE-SSG**) is as follows.

<p align="center">
  <img src="assets/tangent_bounds.gif" alt="Tangent Bound Animation" width="600"/>
  <br>
  <em>Tangent minorizers for the Student's-t likelihood, animated over 50 values of the variational parameter ξ.</em>
</p>

TAVIE-SSG currently supports the following model classes:

| Model class | Description | Examples |
|---|---|---|
| **Type I SSG families** | Linear regression models with heavy-tailed location-scale errors. | Laplace, Student's-t, and general scaled-mixtures of Gaussians. |
| **Type II SSG families** | Count-response generalized linear models. | Binomial/Bernoulli logistic regression and Negative-Binomial regression. |
| **Bayesian Quantile Regression** | Extension of Type I SSG modeling using the asymmetric Laplace likelihood. | Bayesian quantile regression. |

---

## Dependencies

The codebase was developed using:

```text
Python == 3.13.5
```

The package versions below describe the tested environment. Exact version pins are recommended for reproducibility, although nearby compatible versions may also work.

### Core dependencies

These packages are required to run the main TAVIE-SSG implementation and examples:

```text
ipython == 8.30.0, ipykernel == 6.29.5, jupyterlab == 4.3.4 # for running Jupyter notebook
matplotlib == 3.10.0
numpy == 2.4.3
pandas == 2.2.3
rich == 13.9.4
scikit-learn == 1.8.0
scipy == 1.15.3
tqdm == 4.67.1
```

### Additional dependencies for simulations, real-data studies, and competing methods

These packages are required for reproducing the full set of experiments reported in the main manuscript and Supplementary Materials:

```text
arviz == 0.22.0
jax == 0.7.0
jaxlib == 0.7.0
pymc == 5.25.1
pytensor == 2.31.7
rdata == 0.11.2
seaborn == 0.13.2
statsmodels == 0.14.4
xarray == 2025.4.0
torch == 2.7.1
dill == 0.3.8
```

### Built-in Python modules

The following modules are part of the `Python` standard library:

```text
os
re
math
time
random
pickle
logging
warnings
itertools
pathlib
functools
json
typing
dataclasses
```

> **`Python`-version note for `dadvi` and `PyMC (NUTS)`:** The main `TAVIE` module and standalone `PyMC (NUTS)` runs can use `Python 3.13.15` with `PyMC 5.25.1` and `PyTensor 2.31.7`. However, `dadvi` relies on the older `pymc.sampling_jax` API and must be run in a separate `Python 3.9` environment. For installing and setting up `dadvi`, please follow the official instructions at [https://github.com/martiningram/dadvi](https://github.com/martiningram/dadvi).

---

## Reproducibility Guide

Before presenting the TAVIE-SSG functionalities, we list the Jupyter notebooks used to generate the results reported in the main manuscript and Supplementary Materials. Each notebook is linked to its corresponding output notebook, results folder, or data folder to facilitate reproducibility.

| Notebook file | Purpose | Results notebook / folder |
|---|---|---|
| [`TAVIE-SSG_examples.ipynb`](TAVIE-SSG_examples.ipynb) | Implementation examples for TAVIE-SSG under different Type I and Type II SSG likelihoods. | [`TAVIE-SSG_examples.ipynb`](TAVIE-SSG_examples.ipynb) |
| [`student-t_simulations.ipynb`](student-t_simulations.ipynb) | Simulation studies E1 and E2 (Section 4.1) for the Student's-t Type I SSG likelihood, including comparisons with competing methods. | [`results_compete`](results_compete) |
| [`laplace_simulations.ipynb`](laplace_simulations.ipynb) | Simulation studies E1 and E2 (Section M.1) for the Laplace Type I SSG likelihood, including comparisons with competing methods. | [`results_compete`](results_compete) |
| [`negative-binomial_simulations.ipynb`](negative-binomial_simulations.ipynb) | Simulation studies E1 and E2 (Section M.2) for the Negative-Binomial Type II SSG likelihood, including comparisons with competing methods. | [`results_compete`](results_compete) |
| [`STARmap.ipynb`](STARmap.ipynb) | Real-data application to STARmap spatial transcriptomics gene-expression data using the Negative-Binomial TAVIE-SSG model (Section 4.3). | [`data/STARmap`](data/STARmap), [`results_data_study/STARmap`](results_data_study/STARmap) |
| [`BQR_census.ipynb`](BQR_census.ipynb) | Bayesian quantile regression application on the U.S. Census 2000 dataset (Section 4.2), with comparisons against quantile regression, variational, and Monte Carlo competitors. | [`data/Census_data`](data/Census_data), [`results_data_study/census_QR`](results_data_study/census_QR) |
| [`Section_C_gaps.ipynb`](Section_C_gaps.ipynb) & [`Section_C_grid_experiments.ipynb`](Section_C_grid_experiments.ipynb) | Supplementary Section C experiments analyzing ELBO and variational gaps in TAVIE-SSG. | [`results_gaps`](results_gaps) |
| [`Sections_E_F_alpha_sensitivity_calibration.ipynb`](Sections_E_F_alpha_sensitivity_calibration.ipynb) | Supplementary Sections E and F experiments on likelihood-tempering sensitivity and $\alpha$ calibration. | [`results_sensitivity_calibration_alpha`](results_sensitivity_calibration_alpha) |
| [`Section_D_time_complexity_analysis.ipynb`](Section_D_time_complexity_analysis.ipynb) | Supplementary Section D empirical time-complexity analysis of TAVIE-SSG. | [`results_time_complexity`](results_time_complexity) |
| [`Section_J_theoretical_validation.ipynb`](Section_J_theoretical_validation.ipynb) | Supplementary Section J empirical validation of the variational risk bound. | [`results_theory`](results_theory) |
| [`Section_M_3_simulation_alpha.ipynb`](Section_M_3_simulation_alpha.ipynb) | Supplementary Section M.3 simulation study examining the effect of varying $\alpha$. | [`results_alpha`](results_alpha) |
| [`Section_M_4_scaling_p.ipynb`](Section_M_4_scaling_p.ipynb) | Supplementary Section M.4 scaling study with increasing dimension $p$. | [`results_scaling_p`](results_scaling_p) |
| [`TAVIE_SSG_model_misspecification.ipynb`](TAVIE_SSG_model_misspecification.ipynb) | Predictive MSE comparison of TAVIE-SSG and PyMC NUTS under five Type-I Laplace/Student-t misspecification settings and one Type-II Poisson-to-Negative-Binomial setting. | [`results_misspecified`](results_misspecified) |

---

## TAVIE-SSG Functionalities

The `TAVIE` package, located in the `TAVIE/` folder, implements TAVIE-SSG for several strongly super-Gaussian probability models. The implementation is modular and supports different likelihood families, prior structures, and modeling tasks.

### Main TAVIE classes

| Class | Target SSG model | Supported likelihoods | Prior type |
|---|---|---|---|
| `TAVIE_loc_scale` | Heavy-tailed location-scale regression; Type I SSG | Laplace, Student's-t, custom location-scale distributions | Gaussian $\times$ Gamma |
| `TAVIE_type_II` | Count-response GLMs; Type II SSG | Binomial, Negative-Binomial | Gaussian |
| `TAVIE_QR` | Bayesian quantile regression | Asymmetric Laplace | Gaussian |

---

## Importing and Initializing TAVIE Classes

```python
# Import all TAVIE classes
from TAVIE import *

# ------------------------------------------------------------
# Location-scale model: Type I SSG
# Options: family="laplace", family="student", or family="loc_scale"
# ------------------------------------------------------------
loc_scale_model = TAVIE_loc_scale(
    fit_intercept=True,
    scale_X=False,
    scale_y=False,
    family="laplace",
    afunc=None,
    cfunc=None
)

# ------------------------------------------------------------
# Bayesian quantile regression model
# ------------------------------------------------------------
qr_model = TAVIE_QR(
    fit_intercept=True,
    scale_X=False,
    scale_y=False
)

# ------------------------------------------------------------
# Type II SSG model
# Options: family="negbin" or family="binomial"
# ------------------------------------------------------------
type_II_model = TAVIE_type_II(
    fit_intercept=True,
    scale_X=False,
    family="negbin"
)
```

> **Note.** For custom location-scale error distributions using `family="loc_scale"`, the user must provide callable functions `afunc` and `cfunc`.

---

## Custom Location-Scale Families

For a regression problem with general location-scale error distribution:

$$
y_i = \mathbf{x}_i^\top \boldsymbol{\beta} + \epsilon_i, \quad i\in [n],
$$

where the scaled error $\tau \epsilon_i$ has density $p(x)$. The callable functions `afunc` and `cfunc` are defined using $p(x)$ and its derivative $p'(x)$.
Specifically:

$$
A(x) = -\frac{p'(x)}{2x\cdot p(x)} = (2x)^{-1}\cdot \frac{d}{dx}\log p(x),
$$

and:

$$
c(x) = \log p(x) - \frac{x}{2}\cdot \frac{p'(x)}{p(x)} = \log p(x) - \frac{x}{2}\cdot \frac{d}{dx}\log p(x).
$$

For the built-in options `family="laplace"` and `family="student"`, these functions are already implemented internally.

---

## Common Methods Across TAVIE Classes

Each TAVIE class provides a consistent set of methods for fitting, extracting estimates, and tracking convergence.

| Method | `TAVIE_loc_scale` | `TAVIE_type_II` | `TAVIE_QR` | Description |
|---|:---:|:---:|:---:|---|
| `fit()` | ✅ | ✅ | ✅ | Fits the TAVIE-SSG model. |
| `get_TAVIE_means()` | ✅ | ✅ | ✅ | Returns and optionally displays the TAVIE-SSG posterior means. |
| `get_variational_estimates()` | ✅ | ✅ | ✅ | Returns variational estimates of the model-specific hyperparameters. |
| `get_elbo()` | ✅ | ✅ | ✅ | Returns the ELBO trajectory across iterations. |

---

## Example: Type I SSG Laplace Regression

We illustrate TAVIE-SSG using the Type I SSG Laplace regression model.

### Model

The Laplace regression model is:

$$
y_i = \beta_0 + \mathbf{x}_i\boldsymbol{\beta} + \epsilon_i,\quad \epsilon_i \sim \mathrm{Laplace}(0, \sigma=\tau^{-1}),\quad i\in [n],
$$

with $\epsilon_i$ having density (iid for all $i\in [n]$):

$$
f(\epsilon\mid \tau) = \frac{\tau}{2}\exp\left(-\tau \cdot |\epsilon|\right).
$$

The prior is:

$$
\boldsymbol{\beta}\mid \tau^{2}\sim \mathrm{N}(\mathbf{m}, \mathbf{V}/\tau^{2}),\quad \tau^{2}\sim \mathrm{Gamma}(a/2, b/2).
$$

### Simulate data

We generate data with:

$$
(n, p, \tau^{2}_{\mathrm{true}}) = (10^4, 5, 8).
$$

The design matrix $\mathbf{X}\in \mathbb{R}^{n\times \overline{p+1}}$ has standard normal entries. An intercept column $\mathbf{1}_n$ is added automatically by the TAVIE class when `fit_intercept=True`. Also, $\beta_0$ is the intercept, $\boldsymbol{\beta}_{\mathrm{true}} \in \mathbb{R}^{p}$ is generated from a standard normal distribution, and $\epsilon_i\sim \mathrm{Laplace}(0, \tau_\mathrm{true}^{-1})$.

```python
# Simulated data
n = 10000
p = 5
tau2 = 8

# Design matrix
X = np.random.normal(size=(n, p))

# True regression coefficients
beta_true = np.random.normal(loc=0.0, scale=1.0, size=p + 1)

# Laplace noise
error = np.random.laplace(
    loc=0.0,
    scale=1 / np.sqrt(tau2),
    size=n
)

# Response
y = beta_true[0] + X @ beta_true[1:] + error
```

### Fit the TAVIE-SSG model

```python
# Initialize the TAVIE model for the Laplace likelihood
laplace_model = TAVIE_loc_scale(
    family="laplace",
    fit_intercept=True
)

# Fit the model
laplace_model.fit(X, y, verbose=True)
```

### Obtain posterior mean estimates

```python
laplace_model.get_TAVIE_means(verbose=True)
```

<p align="center">
  <img src="assets/laplace_TAVIE_res.png" alt="TAVIE Laplace results" width="600"/>
  <br>
  <em>TAVIE posterior mean estimates for the Laplace regression model.</em>
</p>

### Extract variational estimates

The variational hyperparameter estimates can be obtained as follows:

```python
variational_est = laplace_model.get_variational_estimates()
```

### Extract ELBO trajectory

The ELBO history across iterations can be obtained using:

```python
ELBO = laplace_model.get_elbo()
```

<p align="center">
  <img src="assets/TAVIE_Laplace_ELBO_animation.gif" alt="TAVIE Laplace ELBO" width="600"/>
  <br>
  <em>ELBO trajectory for the Laplace example. TAVIE converged in 112 iterations.</em>
</p>

---

## Additional Examples

Additional examples for other SSG likelihoods and all TAVIE class utilities are provided in:

[`TAVIE-SSG_examples.ipynb`](./TAVIE-SSG_examples.ipynb)

---

## Suggested Reproducibility Workflow for Reviewers

To reproduce the main functionality and examples:

```bash
# 1. Install core dependencies
pip install numpy pandas scipy matplotlib scikit-learn ipython rich tqdm

# 2. Install additional dependencies for full simulations and competing methods
pip install seaborn pymc pytensor arviz jax jaxlib rdata statsmodels xarray

# 3. Install DADVI separately following its official instructions
# https://github.com/martiningram/dadvi
```

Then run the notebooks in the following order:

1. `TAVIE-SSG_examples.ipynb`
2. `student-t_simulations.ipynb`
3. `laplace_simulations.ipynb`
4. `negative-binomial_simulations.ipynb`
5. `STARmap.ipynb`
6. `BQR_census.ipynb`
7. Supplementary notebooks listed in the reproducibility table above.

For a quick check of the core implementation, start with:

```text
TAVIE-SSG_examples.ipynb
```

For reproducing manuscript-level simulation figures and tables, use:

```text
student-t_simulations.ipynb
laplace_simulations.ipynb
negative-binomial_simulations.ipynb
```

For reproducing real-data studies, use:

```text
STARmap.ipynb
BQR_census.ipynb
```

---

## Data studies

We evaluate TAVIE-SSG on two real-data applications.

### U.S. Census 2000: Bayesian quantile regression

We use the U.S. Census 2000 data, containing demographic records from a $5\%$ sample of the U.S. population. The response is the census outcome variable `census_y` (income), and the design matrix `census_X` contains an intercept and demographic covariates: `Female`, age-group indicators (`Age30`, `Age40`, `Age50`, `Age60`, `Age70`), `NonWhite`, `Married`, `Education`, and `Education2`; $n=5\times 10^6$ and $p=11$. We fit Bayesian quantile regression under the Asymmetric Laplace likelihood across quantile levels $(0.05, 0.10, \ldots, 0.95)$. The proposed TAVIE-SSG framework is compared with DADVI, ADVI (MF), ADVI (FR), PyMC (NUTS), FAST QR, and statsmodels. See [`BQR_census.ipynb`](BQR_census.ipynb).

### STARmap spatial transcriptomics: Negative-Binomial regression

We use the STARmap visual cortex spatial transcriptomics dataset, focusing on one representative tissue sample with $G=160$ genes measured over $n=941$ spatial locations. The response is the gene-expression count matrix, where each column corresponds to one gene and each row corresponds to one spatial location. Spatial coordinates are used to construct spline-based covariates ($p=27$), and each gene is modeled separately using a Negative-Binomial regression model. The proposed TAVIE-SSG framework is compared with DADVI, ADVI (MF), ADVI (FR), and PyMC (NUTS) in terms of fitted gene-expression patterns, prediction quality, residual diagnostics, and runtime. See [`STARmap.ipynb`](STARmap.ipynb).
