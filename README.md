# BAB-OPDE

## Bayesian Adaptive B-Spline-based Optimal Probability Density Estimation for Mineral Flotation Bubble Size and Morphology Monitoring

[![Status](https://img.shields.io/badge/Status-Published-brightgreen.svg)](https://doi.org/10.1109/TASE.2026.3731356)
[![Journal](https://img.shields.io/badge/IEEE-TASE-blue.svg)](https://doi.org/10.1109/TASE.2026.3731356)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTASE.2026.3731356-blue.svg)](https://doi.org/10.1109/TASE.2026.3731356)

Official repository for the paper:

> **J. Liu, H. Lan, D. Luo, H. Shao, Z. Tang, and Y. Xie**,  
> “Bayesian Adaptive B-Spline-based Optimal Probability Density Estimation for Mineral Flotation Bubble Size and Morphology Monitoring,”  
> *IEEE Transactions on Automation Science and Engineering*, 2026.  
> DOI: [10.1109/TASE.2026.3731356](https://doi.org/10.1109/TASE.2026.3731356)

---

## Overview

Reliable monitoring of bubble size and morphological characteristics (BSMCs) is important for froth flotation process monitoring and control. Image-level BSMC distributions contain richer process-state information than simple statistics such as the mean and variance, but they are often affected by segmentation errors, local froth disturbances, and process noise.

B-spline-based probability density function estimation (PDFE) represents a BSMC distribution using a finite-dimensional B-spline weight vector. This representation is compact and interpretable, but fixed-basis B-spline PDFE may produce oscillatory density estimates and unstable weight features when local fluctuations are overfitted.

To address these issues, this work proposes **Bayesian Adaptive B-spline-based Optimal Probability Density Estimation (BAB-OPDE)**. BAB-OPDE introduces a Bayesian smoothing prior to regularize neighboring B-spline weights, uses variational inference to estimate the weights and uncertainty-related parameters, and adaptively removes low-contribution basis functions to obtain a compact representation.

The method is intended to preserve dominant long-tailed, multimodal, and joint distributional structures of flotation BSMCs while suppressing unnecessary local fluctuations.

---

## BSMC Extraction

Froth image segmentation is used to isolate individual bubbles and extract bubble size and morphological characteristics. The following example shows representative segmentation and BSMC measurement results obtained using the MsD-MsJ-based froth image segmentation method.

<div align="center">
  <img width="90%" alt="Froth image segmentation and BSMC extraction" src="https://github.com/user-attachments/assets/6da9c616-1c17-4f79-9562-98976cf134a3" />
</div>

---

## Method Highlights

BAB-OPDE mainly includes the following components:

1. **B-spline weight-based PDF representation**  
   The BSMC distribution is represented as a weighted combination of B-spline basis functions, converting PDF estimation into finite-dimensional B-spline weight estimation under valid PDF constraints.

2. **Bayesian smoothing prior**  
   A prior smoothness constraint regularizes neighboring B-spline weights, reducing sensitivity to segmentation errors, local froth disturbances, and process noise.

3. **Variational Bayesian inference**  
   Variational inference is used to estimate the B-spline weights together with uncertainty-related parameters.

4. **Adaptive basis selection**  
   Low-contribution basis functions are progressively removed to obtain a compact and stable B-spline representation under different flotation conditions.

The formulation can be applied to both one-dimensional BSMC distributions and multidimensional joint distributions.

---

## Experimental Results

### 1. Numerical PDF Estimation

BAB-OPDE was evaluated on one-dimensional and two-dimensional synthetic distributions containing multimodal and non-Gaussian structures.

For the two-dimensional task, fixed-basis B-spline PDFE captures the overall density trend but exhibits visible oscillations around multimodal regions. KDE produces a smoother surface but may blur local multimodal structures. BAB-OPDE preserves the dominant mode structure while reducing unnecessary oscillations in the B-spline density estimate.

<div align="center">
  <img width="100%" alt="Two-dimensional PDF estimation results" src="https://github.com/user-attachments/assets/db38d56d-c404-4e66-b7c2-9c415804733e" />
</div>

Under the default two-dimensional setting, BAB-OPDE reduces the number of active B-spline basis functions from **900 to 440** while maintaining competitive density-estimation accuracy and smoothness. Compared with traditional fixed-basis B-spline PDFE, the MSE decreases from **2.19 × 10⁻⁵** to **1.47 × 10⁻⁵**, and the MAE decreases from **2.14 × 10⁻³** to **1.79 × 10⁻³**.

### 2. Convergence Analysis

The variational inference and adaptive basis-selection procedure exhibits stage-wise convergence. Within each stage, the variational objective approaches a stable value, while the active basis set is updated between stages.

<div align="center">
  <img width="55%" alt="BAB-OPDE convergence analysis" src="https://github.com/user-attachments/assets/2cd51319-e3ab-42e4-a2d8-a384094410c1" />
</div>

### 3. Copper Flotation Case Study

BAB-OPDE was also evaluated using real copper flotation data. The method was used to estimate distributions of BSMCs such as bubble size, eccentricity, and orientation, including their joint distributions.

The following example shows the temporal evolution of the joint distribution of bubble eccentricity and orientation together with representative froth images.

<div align="center">
  <img width="100%" alt="Joint distribution evolution in the copper flotation process" src="https://github.com/user-attachments/assets/b3731000-b072-4b38-96f1-fe4c76aa1688" />
</div>

### 4. Flotation Condition Monitoring

The estimated B-spline weights were further evaluated as distribution-level features for a downstream five-class flotation condition monitoring task.

Among the compared BSMC representations, the BAB-OPDE representation achieved the highest **Accuracy** and **Macro-F1**. The ablation experiment without the Bayesian smoothing prior showed a clear performance decrease, supporting the importance of suppressing noise-driven fluctuations in the B-spline weight features.

> **Data availability note:** the copper flotation experiments use proprietary industrial data. These data are not included in this repository.

---

## Repository Contents

This repository provides the following public files:

```text
BAB-OPDE/
├── README.md
├── babopde.py
└── data.csv
```

### `babopde.py`

`babopde.py` provides the Python implementation of BAB-OPDE, including:

- B-spline basis construction;
- fixed-point maximum-likelihood initialization;
- variational inference for the B-spline weights and variance-related parameters;
- multidimensional second-order smoothness regularization;
- adaptive pruning of low-contribution basis functions;
- PDF evaluation on arbitrary samples or grids; and
- convergence-history recording.

### `data.csv`

`data.csv` contains **3,000 two-dimensional synthetic samples** with two columns:

```text
x1, x2
```

---

## Quick Start

### Requirements

The implementation requires:

```text
numpy
scipy
```

Install the dependencies with:

```bash
pip install numpy scipy
```

### Example

```python
import numpy as np
from babopde import BABOPDE

# Load the released synthetic data
X = np.loadtxt("data.csv", delimiter=",", skiprows=1)

# A relatively small basis size can be used for a quick test
model = BABOPDE(
    degree=3,
    alpha0=1,
    beta0=1,
    eps0=1,
    eta0=1,
    vi_tol=1e-5,
    outer_tol=1e-3,
    prune_tol=0.05,
)

model.fit(
    X,
    n_basis_init=[12, 12],
)

# Evaluate the estimated PDF on a regular grid
r1, r2 = model.ranges_
g1 = np.linspace(r1[0], r1[1], 100)
g2 = np.linspace(r2[0], r2[1], 100)

mesh, pdf = model.evaluate_grid([g1, g2])

print("Final active basis shape:", model.coef_shape_)
print("Number of active coefficients:", len(model.coef_))
```

For the main two-dimensional numerical experiment reported in the paper, the initial basis size is **30 × 30** and the pruning threshold is **λ = 0.05**.

---

## Citation

If you use BAB-OPDE in your research, please cite:

```bibtex
@article{Liu2026BABOPDE,
  author  = {J. Liu and H. Lan and D. Luo and H. Shao and Z. Tang and Y. Xie},
  title   = {Bayesian Adaptive B-Spline-based Optimal Probability Density Estimation for Mineral Flotation Bubble Size and Morphology Monitoring},
  journal = {IEEE Transactions on Automation Science and Engineering},
  year    = {2026},
  doi     = {10.1109/TASE.2026.3731356}
}
```

---

## Usage Notice

This repository is intended for **academic research and non-commercial use**.

Redistribution of the code or data without permission is prohibited. For other usage scenarios, please contact the authors.

---

## Contact

For questions about the paper or implementation, please contact:

**Email:** lhx.contact@foxmail.com
