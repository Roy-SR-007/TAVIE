<h1 align="center">TAVIE-SSG</h1>

<p align="center">
  <b>A Generalized Tangent Approximation based Variational Inference Framework<br>for Strongly Super-Gaussian Likelihoods</b>
</p>

<p align="center">
  Somjit Roy<sup>1</sup> &nbsp;&middot;&nbsp;
  Pritam Dey<sup>1</sup> &nbsp;&middot;&nbsp;
  Debdeep Pati<sup>2</sup> &nbsp;&middot;&nbsp;
  Bani K. Mallick<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> <em>Department of Statistics, Texas A&amp;M University</em><br>
  <sup>2</sup> <em>Department of Statistics, University of Wisconsin-Madison</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2504.05431"><b>📄 Read the paper on arXiv</b></a>
</p>

<p align="center">
  <img src="assets/TAVIE-SSG_animated_logo.gif" alt="TAVIE-SSG logo" width="650"/>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2504.05431"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2504.05431-B31B1B?logo=arxiv&logoColor=white"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%20%7C%203.12-3776AB?logo=python&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
  <img alt="PyMC" src="https://img.shields.io/badge/PyMC-5.25-1E4C7E">
  <img alt="JAX" src="https://img.shields.io/badge/JAX-x64-8A2BE2">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

<p align="center">
  <a href="https://github.com/Roy-SR-007/TAVIE-SSG/fork"><img alt="Forks" src="https://img.shields.io/github/forks/Roy-SR-007/TAVIE-SSG?logo=github&label=forks&color=4C7EE0"></a>
  <a href="https://github.com/Roy-SR-007/TAVIE-SSG/releases"><img alt="Downloads" src="https://img.shields.io/github/downloads/Roy-SR-007/TAVIE-SSG/total?logo=github&label=downloads&color=4C7EE0"></a>
  <a href="https://github.com/Roy-SR-007/TAVIE-SSG/commits/main"><img alt="Last updated" src="https://img.shields.io/github/last-commit/Roy-SR-007/TAVIE-SSG/main?logo=github&label=last%20updated&color=4C7EE0"></a>
</p>

<p align="center">
  <i>Source code, data, and a complete reproduction pipeline for the implementation of TAVIE-SSG</i>
</p>

---

## Contents

| | Section | For |
|:--|:--|:--|
| 1 | [Overview](#1-overview) | what TAVIE-SSG is and what it supports |
| 2 | [Repository map](#2-repository-map) | finding your way around the folder |
| 3 | [Installation and environments](#3-installation-and-environments) | setting up before running anything |
| 4 | [Reproducibility guide](#4-reproducibility-guide) | users reproducing results in the paper |
| 5 | [Using TAVIE-SSG](#5-using-tavie-ssg) | the API |
| 6 | [Worked example: Laplace regression](#6-worked-example-laplace-regression) | a five-minute tour |
| 7 | [Real-data studies](#7-real-data-studies) | the two applications |
| 8 | [Citation and license](#8-citation-and-license) | citing the paper and reuse terms |

---

## 1. Overview

**T**angent **A**pproximation based **V**ariational **I**nferenc**E** for **S**trongly
**S**uper-**G**aussian likelihoods (**TAVIE-SSG**) builds a variational posterior from
tangent minorizers of the likelihood, giving closed-form coordinate updates and a
monotone ELBO for a broad class of heavy-tailed and discrete/count-response models.

<p align="center">
  <img src="assets/tangent_bounds.gif" alt="Tangent bound animation" width="600"/>
  <br>
  <em>Tangent minorizers for the Student's-t likelihood, animated over 50 values of the variational parameter ξ.</em>
</p>

### Supported model classes

| Model class | Description | Examples | TAVIE class |
|:--|:--|:--|:--|
| **Type I SSG** | Linear regression with heavy-tailed location-scale errors | Laplace, Student's-t, general scale mixtures of Gaussians | `TAVIE_loc_scale` |
| **Type II SSG** | Discrete/Count-response generalized linear models | Binomial/Bernoulli logistic, Negative-Binomial | `TAVIE_type_II` |
| **Bayesian quantile regression** | Type I SSG under the asymmetric Laplace likelihood | Quantile regression at any level τ | `TAVIE_QR` |

---

## 2. Repository map

Every path in every notebook is **relative to the repository root**, so launch Jupyter
from this directory.

```text
TAVIE-SSG/
│
├── TAVIE/                    ← the package. Import with `from TAVIE import *`
│   ├── tavie.py                  core variational updates
│   ├── TAVIE_classes.py          TAVIE_loc_scale, TAVIE_type_II, TAVIE_QR
│   └── utils.py
│
├── TAVIE_true_elbo/          ← instrumented variant of the same package
│   ├── compare_true_elbo.py      compare_type_I / compare_type_II
│   ├── tavie_with_history.py     records the full ELBO trajectory
│   └── ...                       (otherwise mirrors TAVIE/)
│
├── CompetingMethods/         ← the baselines TAVIE-SSG is compared against
│   ├── bbvi.py                   ADVI mean-field and full-rank
│   ├── bbvi_qr.py                ADVI for quantile regression
│   ├── mfvi.py                   mean-field VI
│   └── logit_CAVI.py, logit_SVI.py
│
├── dadvi/                    ← vendored DADVI (Ingram et al.), incl. its PyMC/JAX API
│
├── data/                     ← input data, shipped with the repository
│   ├── Census_data/census_data.mat            33 MB, from the FAST-QR repository
│   └── STARmap/dark_replicate_1.*.csv         counts, gene symbols, locations
│
├── assets/                   ← figures and animations used by this README
│
├── *.ipynb                   ← 14 notebooks; see §4
│
└── results_*/                ← outputs, all shipped populated (see §4.1)
    ├── results_compete/                  Type I/II simulation studies
    ├── results_data_study/               census_QR/ and STARmap/
    ├── results_gaps/                     Section C gap analyses
    ├── results_theory/                   Section J risk bounds
    ├── results_alpha/                    Section M.3 likelihood power
    ├── results_scaling_p/                Section M.4 dimension scaling
    ├── results_time_complexity/          Section D timing
    ├── results_misspecified/             misspecification study
    └── results_sensitivity_calibration_alpha/   Sections E and F
```

> [!NOTE]
> **`TAVIE/` vs `TAVIE_true_elbo/`.** These are *not* interchangeable. `TAVIE/` is the
> package for ordinary use. `TAVIE_true_elbo/` additionally computes the true ELBO by
> Monte Carlo and records per-iteration history; it is imported only by
> `Section_C_gaps.ipynb` and `Section_C_grid_experiments.ipynb`, which study the gap
> between the two objectives.

---

## 3. Installation and environments

### 3.1 Two environments are required

The notebooks were run under two kernels, and the reproduction of some results depends
on the older `pymc.sampling_jax` API that `dadvi` relies on. Create both:

```bash
# Environment A — main environment
conda create -n tavie python=3.13
conda activate tavie
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels \
            pymc pytensor arviz xarray jax jaxlib rich tqdm dill torch \
            ipython ipykernel jupyterlab

# Environment B — for the notebooks that call DADVI
conda create -n dadvi python=3.9
conda activate dadvi
# then follow the official DADVI instructions:
# https://github.com/martiningram/dadvi
```

Both environments need the local packages (`TAVIE`, `TAVIE_true_elbo`,
`CompetingMethods`, `dadvi`) importable — they live in this repository and are picked up
automatically when Jupyter is launched from the repository root.

### 3.2 Tested package versions

Exact pins reproduce the reported numbers; nearby versions will usually work.

**Core** — required for the TAVIE-SSG implementation and the examples:

```text
ipython == 8.30.0, ipykernel == 6.29.5, jupyterlab == 4.3.4
matplotlib == 3.10.0
numpy == 2.4.3
pandas == 2.2.3
rich == 13.9.4
scikit-learn == 1.8.0
scipy == 1.15.3
tqdm == 4.67.1
```

**Additional** — required for the simulations, real-data studies, and competing methods:

```text
arviz == 0.22.0
jax == 0.7.0
jaxlib == 0.7.0
pymc == 5.25.1
pytensor == 2.31.7
seaborn == 0.13.2
statsmodels == 0.14.4
xarray == 2025.4.0
torch == 2.7.1
dill == 0.3.8
```

**Standard library** (no installation needed): `os`, `re`, `math`, `time`, `random`,
`pickle`, `logging`, `warnings`, `itertools`, `pathlib`, `functools`, `json`, `typing`,
`dataclasses`.

> [!IMPORTANT]
> **DADVI and PyMC versions.** The main `TAVIE` module and standalone PyMC (NUTS) runs
> work under Python 3.13.5 with PyMC 5.25.1 and PyTensor 2.31.7. `dadvi` relies on the
> older `pymc.sampling_jax` API; install and configure it per the official instructions
> at [github.com/martiningram/dadvi](https://github.com/martiningram/dadvi).

---

## 4. Reproducibility guide

### 4.1 Two entry points

**Every `results_*` directory in this repository ships populated** — both the fitted
result files and the final figures. That gives two ways to reproduce:

| | Path | What you run |
|:--|:--|:--|
| **A** | **Regenerate figures only** | Skip the fitting cells; run the plotting cells, which read the shipped `.pkl` / `.csv` files from disk |
| **B** | **Refit from scratch** | Run every cell top to bottom; the fitting cells overwrite the shipped result files |

> [!IMPORTANT]
> **Within a notebook, always run cells in order, top to bottom for full reproduction of the results in the paper.**

### 4.2 Notebook index

| Notebook | Purpose | Outputs to |
|:--|:--|:--|
| [`TAVIE-SSG_examples.ipynb`](TAVIE-SSG_examples.ipynb) | Worked examples of every Type I and Type II SSG likelihood and the full class API. **Start here.** | *(inline only)* |
| [`student-t_simulations.ipynb`](student-t_simulations.ipynb) | Experiments E1 and E2 (Section 4.1), Student's-t Type I, against all competitors | [`results_compete`](results_compete) |
| [`laplace_simulations.ipynb`](laplace_simulations.ipynb) | Experiments E1 and E2 (Section M.1), Laplace Type I | [`results_compete`](results_compete) |
| [`negative-binomial_simulations.ipynb`](negative-binomial_simulations.ipynb) | Experiments E1 and E2 (Section M.2), Negative-Binomial Type II | [`results_compete`](results_compete) |
| [`BQR_census.ipynb`](BQR_census.ipynb) | Bayesian quantile regression on U.S. Census 2000 (Section 4.2) | [`data/Census_data`](data/Census_data) → [`results_data_study/census_QR`](results_data_study/census_QR) |
| [`STARmap.ipynb`](STARmap.ipynb) | Spatial transcriptomics, Negative-Binomial (Section 4.3) | [`data/STARmap`](data/STARmap) → [`results_data_study/STARmap`](results_data_study/STARmap) |
| [`Section_C_gaps.ipynb`](Section_C_gaps.ipynb) | Jensen's gap, ELBO gap, posterior concentration (Sections C.1, C.2, C.4, C.5) | [`results_gaps`](results_gaps) |
| [`Section_C_grid_experiments.ipynb`](Section_C_grid_experiments.ipynb) | Student's-t grid: one TAVIE fit per job, all gap quantities | [`results_gaps`](results_gaps) |
| [`Section_D_time_complexity_analysis.ipynb`](Section_D_time_complexity_analysis.ipynb) | Empirical time complexity in *n* and *p* (Section D) | [`results_time_complexity`](results_time_complexity) |
| [`Sections_E_F_alpha_sensitivity_calibration.ipynb`](Sections_E_F_alpha_sensitivity_calibration.ipynb) | Sensitivity to α, and the calibration algorithm (Section 2.4; Sections E and F) | [`results_sensitivity_calibration_alpha`](results_sensitivity_calibration_alpha) |
| [`Section_J_theoretical_validation.ipynb`](Section_J_theoretical_validation.ipynb) | Empirical validation of the variational risk bounds (Section J) | [`results_theory`](results_theory) |
| [`Section_M_3_simulation_alpha.ipynb`](Section_M_3_simulation_alpha.ipynb) | Effect of the likelihood power α (Section M.3) | [`results_alpha`](results_alpha) |
| [`Section_M_4_scaling_p.ipynb`](Section_M_4_scaling_p.ipynb) | Scaling with dimension *p* (Section M.4) | [`results_scaling_p`](results_scaling_p) |
| [`TAVIE_SSG_model_misspecification.ipynb`](TAVIE_SSG_model_misspecification.ipynb) | Predictive MSE under six misspecified likelihoods | [`results_misspecified`](results_misspecified) |

### 4.3 Figure index — Main Manuscript

| Figure | Notebook | Output file |
|:--|:--|:--|
| **1(a), 1(b)** | `student-t_simulations` | `results_compete/plots/runtime_Student_multi_n_multi_p.pdf` |
| **1(c)** | `BQR_census` | `results_data_study/census_QR/runtime_comparison_n_10000.pdf` |
| **2(a)** | `Section_C_gaps` | `results_gaps/plots/student_t_tangent_plot.pdf` |
| **2(b)** | `Section_C_gaps` | `results_gaps/plots/student_t_jensen_gap_xi1star_zeta1star.pdf` |
| **2(c)** | `Section_C_gaps` | `results_gaps/plots/nb_tangent_plot.pdf` |
| **2(d)** | `Section_C_gaps` | `results_gaps/plots/negbin_jensen_gap_xi1star_zeta1star.pdf` |
| **4** | `student-t_simulations` | `results_compete/plots/Student_SW_MSE_boxplots_multi_n_p_8_1x3.pdf` |
| **5** | `student-t_simulations` | `results_compete/plots/Student_MSE_boxplots_multi_p_n_1000.pdf` |
| **6(a)** | `BQR_census` | `results_data_study/census_QR/heatmap_tavie_QR_FAST_QR_differences.pdf` |
| **6(b)** | `BQR_census` | `results_data_study/census_QR/census_estimates_competing_methods_n_10000_selected.pdf` |
| **7(b)** | `STARmap` | `results_data_study/STARmap/Pcp4.jpeg` |
| **8(a)** | `STARmap` | `results_data_study/STARmap/Slc17a7.jpeg` |

### 4.4 Figure index — Supplementary Materials

| Figure | Notebook | Output file *(under the notebook's results directory)* |
|:--|:--|:--|
| **1** (left, right) | `Section_C_gaps` | `plots/student_t_jensen_gap_highlighted.pdf`, `plots/nb_jensen_gap_highlighted.pdf` |
| **2** (left, right) | `Section_C_gaps` | `plots/student_t_jensen_gap_xi1star_zeta1star.pdf`, `plots/negbin_jensen_gap_xi1star_zeta1star.pdf` |
| **3** | `Section_C_gaps` | `plots/tavie_true_elbo_gap_comparison.pdf` |
| **4** | `Section_C_gaps` | `plots/student_t_boxplot_postconv_elbo_gap_fix_n_vary_p.pdf` |
| **5(a)–(d)** | `Section_C_grid_experiments` | `plots/gap_vs_p_linear_fit.pdf`, `plots/gap_vs_log_n.pdf`, `plots/gap_vs_half_p_log_n_linear_fit.pdf`, `plots/log_gap_vs_log_half_p_log_n.pdf` |
| **6** | `Section_C_grid_experiments` | `plots/student_t_logml_gap_boxplots.pdf` |
| **7(a), 7(b)** | `Section_C_grid_experiments` | `plots/student_t_logml_gap_vs_half_p_log_n_linear_fit.pdf`, `plots/student_t_log_gap_vs_log_half_p_log_n_linear_fit.pdf` |
| **8** | `Section_C_grid_experiments` | `plots/student_t_elbo_minus_true_loglik_vs_half_p_log_n_linear_fit.pdf` |
| **9** | `Section_C_gaps` | `contour_plots/student_side_by_side_all_methods_n_1000_alpha_1.00_replot.pdf`, `…_n_2000_…pdf` |
| **10** | `student-t_simulations` | `plots/Student_SW_boxplots_multi_n_p_8_logscale.pdf` |
| **11** | `Section_D_time_complexity_analysis` | `tavie_scaling.pdf` |
| **12** | `Sections_E_F` | `contours_beta1_beta2.pdf` |
| **13, 14** | `Sections_E_F` | `boxplot_mse_beta_alpha_laplace.pdf`, `boxplot_mse_tau2_alpha_laplace.pdf`, `boxplot_renyi_alpha_laplace.pdf` |
| **15** | `Sections_E_F` | `coverage_overall_vs_alpha.pdf`, `width_overall_vs_alpha.pdf` |
| **16** | `Sections_E_F` | `coverage_boxplots_by_n_alpha.pdf` |
| **17** | `Sections_E_F` | `calibrated_alpha_with_history.pdf`, `algorithm_history.pdf` |
| **18, 19** | `Section_J_theoretical_validation` | `plots/variational_risk_bound_gap_Laplace_n_2000_p_8.pdf`, `…_n_10000_p_8.pdf` |
| **20, 21** | `Section_J_theoretical_validation` | `plots/variational_risk_bound_gap_NB_alpha_less_1_n_2000_p_8.pdf`, `…_n_10000_p_8.pdf` |
| **22** | `student-t_simulations` | `plots/Student_ELBO_n_2000_p_8.pdf` |
| **23** | `laplace_simulations` | `plots/Laplace_MSE_boxplots_multi_n_p_8.pdf` |
| **24** | `laplace_simulations` | `plots/Laplace_SW_boxplots_multi_n_p_8_logscale.pdf` |
| **25** | `laplace_simulations` | `plots/Laplace_MSE_boxplots_multi_p_n_1000.pdf` |
| **26** | `laplace_simulations` | `plots/runtime_Laplace_multi_n_multi_p.pdf` |
| **27** | `laplace_simulations` | `plots/Laplace_ELBO_n_1000_p_8.pdf` |
| **28** | `negative-binomial_simulations` | `plots/Neg-Bin_MSE_boxplots_multi_n_p_8.pdf` |
| **29** | `negative-binomial_simulations` | `plots/Negbin_SW_boxplots_multi_n_p_8_logscale.pdf` |
| **30** | `negative-binomial_simulations` | `plots/Neg-Bin_MSE_boxplots_multi_p_n_1000.pdf` |
| **31** | `negative-binomial_simulations` | `plots/runtime_Negbin_multi_n_multi_p.pdf` |
| **32** | `negative-binomial_simulations` | `plots/NegBin_ELBO_n_1000_p_8.pdf` |
| **33, 34, 35** | `Section_M_3_simulation_alpha` | `plots/Student_alpha_MSE_boxplots.pdf`, `plots/Laplace_alpha_MSE_boxplots.pdf`, `plots/NegBin_alpha_MSE_boxplots.pdf` |
| **36, 37** | `Section_M_4_scaling_p` | `mse_boxplots_scaling_p_student_t.pdf`, `mse_boxplots_scaling_p_negbin.pdf` |
| **38** | `BQR_census` | `tavie_QR_estimates_census_all.pdf` |
| **39** | `BQR_census` | `census_estimates_competing_methods_n_10000_all.pdf` |
| **40** | `BQR_census` | `census_TAVIE_ELBO.pdf` |
| **41** | `BQR_census` | `census_ADVI_FR_ELBO.pdf` |
| **42(a)–(f)** | `STARmap` | `Lmo2.jpeg`, `Chat.jpeg`, `Egr1.jpeg`, `Fam19a1.jpeg`, `Mog.jpeg`, `Gpc3.jpeg` |
| **43(a)–(d)** | `STARmap` | `heatmap_random_genes_1_40.pdf`, `…_41_80.pdf`, `…_81_120.pdf`, `…_121_160.pdf` |
| **44** | `TAVIE_SSG_model_misspecification` | `tavie_ssg_misspecification_six_experiments_mse_boxplots.pdf` |

### 4.5 Expected runtimes

**TAVIE-SSG is fast having a target regime of large $n$ with small to moderate $p$.** The
per-iteration cost grows roughly linearly in $n$ and cubically in $p$ — refer to the empirical
log–log slopes in [`Section_D_time_complexity_analysis.ipynb`](Section_D_time_complexity_analysis.ipynb).

Competitor runtimes in particular are sensitive to machine configuration.

> [!NOTE]
> All experiments reported here were conducted on a **MacBook Air (M2) with 8 GB of RAM**.

### 4.6 A suggested order

```text
1.  TAVIE-SSG_examples.ipynb            confirm the installation works
2.  student-t_simulations.ipynb          Section 4.1  →  Figures 1(a,b), 4, 5, 10, 22
3.  laplace_simulations.ipynb            Section M.1  →  Figures 23–27
4.  negative-binomial_simulations.ipynb  Section M.2  →  Figures 28–32
5.  BQR_census.ipynb                     Section 4.2  →  Figures 1(c), 6, 38–41
6.  STARmap.ipynb                        Section 4.3  →  Figures 7(b), 8(a), 42, 43
7.  the remaining supplementary notebooks, in any order
```

---

## 5. Using TAVIE-SSG

The `TAVIE` package implements TAVIE-SSG for several strongly super-Gaussian models. It
is modular across likelihood families, prior structures, and modeling tasks.

### 5.1 Classes

| Class | Target SSG model | Supported likelihoods | Prior |
|:--|:--|:--|:--|
| `TAVIE_loc_scale` | Heavy-tailed location–scale regression (Type I) | Laplace, Student's-t, custom location–scale | Gaussian $\times$ Gamma |
| `TAVIE_type_II` | Discrete/Count-response GLMs (Type II) | Binomial (includes Logistic), Negative-Binomial | Gaussian |
| `TAVIE_QR` | Bayesian quantile regression | Asymmetric Laplace | Gaussian |

### 5.2 Importing and initializing

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

> [!NOTE]
> For custom location–scale error distributions via `family="loc_scale"`, you must supply
> the callables `afunc` and `cfunc` (see §5.3).

### 5.3 Custom location–scale families

For a regression problem with a general location–scale error distribution,

$$
y_i = \mathbf{x}_i^\top \boldsymbol{\beta} + \epsilon_i, \quad i\in [n],
$$

where the scaled error $\tau \epsilon_i$ has density $p(x)$, the callables `afunc` and
`cfunc` are defined from $p(x)$ and $p'(x)$:

$$
A(x) = -\frac{p'(x)}{2x\cdot p(x)} = (2x)^{-1}\cdot \frac{d}{dx}\log p(x),
$$

$$
c(x) = \log p(x) - \frac{x}{2}\cdot \frac{p'(x)}{p(x)} = \log p(x) - \frac{x}{2}\cdot \frac{d}{dx}\log p(x).
$$

For the built-in `family="laplace"` and `family="student"` these are implemented
internally.

### 5.4 Common methods

Every TAVIE class exposes the same interface for fitting, extracting estimates, and
tracking convergence.

| Method | `TAVIE_loc_scale` | `TAVIE_type_II` | `TAVIE_QR` | Description |
|:--|:---:|:---:|:---:|:--|
| `fit()` | ✅ | ✅ | ✅ | Fits the TAVIE-SSG model |
| `get_TAVIE_means()` | ✅ | ✅ | ✅ | Posterior means, optionally displayed |
| `get_variational_estimates()` | ✅ | ✅ | ✅ | Variational estimates of the model hyperparameters |
| `get_elbo()` | ✅ | ✅ | ✅ | ELBO trajectory across iterations |

---

## 6. Worked example: Laplace regression

A Type I SSG model, end to end.

### 6.1 The model

$$
y_i = \beta_0 + \mathbf{x}_i\boldsymbol{\beta} + \epsilon_i,\quad \epsilon_i \sim \mathrm{Laplace}(0, \sigma=\tau^{-1}),\quad i\in [n],
$$

with errors i.i.d. across $i$,

$$
f(\epsilon\mid \tau) = \frac{\tau}{2}\exp\left(-\tau \cdot |\epsilon|\right),
$$

and prior

$$
\boldsymbol{\beta}\mid \tau^{2}\sim \mathrm{N}(\mathbf{m}, \mathbf{V}/\tau^{2}),\quad \tau^{2}\sim \mathrm{Gamma}(a/2, b/2).
$$

### 6.2 Simulate

We take $(n, p, \tau^{2}_{\mathrm{true}}) = (10^4, 5, 8)$. The design matrix has standard
normal entries; the intercept column is added automatically when `fit_intercept=True`.

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

### 6.3 Fit

```python
laplace_model = TAVIE_loc_scale(
    family="laplace",
    fit_intercept=True
)

laplace_model.fit(X, y, verbose=True)
```

### 6.4 Posterior means

```python
laplace_model.get_TAVIE_means(verbose=True)
```

<p align="center">
  <img src="assets/laplace_TAVIE_res.png" alt="TAVIE Laplace results" width="600"/>
  <br>
  <em>TAVIE-SSG posterior mean estimates for the Laplace regression model.</em>
</p>

### 6.5 Variational estimates and ELBO

```python
variational_est = laplace_model.get_variational_estimates()
ELBO = laplace_model.get_elbo()
```

<p align="center">
  <img src="assets/TAVIE_Laplace_ELBO_animation.gif" alt="TAVIE Laplace ELBO" width="600"/>
  <br>
  <em>ELBO trajectory for the Laplace example. TAVIE-SSG converged in 112 iterations.</em>
</p>

Further examples covering every SSG likelihood and every class utility are in
[`TAVIE-SSG_examples.ipynb`](TAVIE-SSG_examples.ipynb).

---

## 7. Real-data studies

### 7.1 U.S. Census 2000 — Bayesian quantile regression

Demographic records from a 5% sample of the U.S. population, $n = 5\times 10^{6}$ and
$p = 11$. The response `census_y` is log annual salary; the design matrix `census_X`
holds an intercept plus `Female`, the age-bracket indicators `Age30`–`Age70`, `NonWhite`,
`Married`, `Education` and `Education2`. Bayesian quantile regression is fitted under the
asymmetric Laplace likelihood at the 19 levels $\tau = 0.05, 0.10, \ldots, 0.95$, and
TAVIE-SSG is compared against DADVI, ADVI (MF), ADVI (FR), PyMC (NUTS), FAST-QR and
`statsmodels`.

Data ships in [`data/Census_data`](data/Census_data), taken verbatim from the FAST-QR
solver repository of Yang, Meng and Mahoney (2013) so that the reference solution matches
the published one exactly. See [`BQR_census.ipynb`](BQR_census.ipynb).

### 7.2 STARmap spatial transcriptomics — Negative-Binomial regression

One tissue sample from the STARmap visual cortex dataset, with $G = 160$ genes measured
over $n = 941$ spatial locations. Spatial coordinates are expanded into a tensor-product
cubic B-spline basis ($p = 27$), and each gene is modeled separately by Negative-Binomial
regression. TAVIE-SSG is compared against DADVI, ADVI (MF), ADVI (FR) and PyMC (NUTS) on
fitted expression patterns, prediction quality, residual diagnostics and runtime.

Data ships in [`data/STARmap`](data/STARmap). See [`STARmap.ipynb`](STARmap.ipynb).

---

## 8. Citation and license

If you use this code, please cite:

```bibtex
@misc{tavie-ssg,
title={A Generalized Tangent Approximation based Variational Inference Framework for Strongly Super-Gaussian Likelihoods}, 
author={Somjit Roy and Pritam Dey and Debdeep Pati and Bani K. Mallick},
year={2026},
note={arXiv:2504.05431},
url={https://arxiv.org/abs/2504.05431}
}
```

Released under the [MIT License](LICENSE). © 2026 Somjit Roy, Pritam Dey, Debdeep Pati
and Bani K. Mallick.