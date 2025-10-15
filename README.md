# Structured Missingness Simulation

This Python module provides a set of functions for simulating structured missingness mechanisms, building upon the work of Rubin [[1]](#1) as defined by Jackson et al. [[2]](#2). It implements mechanisms under Missing Completely At Random (MCAR), Missing At Random (MAR), and Missing Not At Random (MNAR). The simulation uses a logistic masking model to introduce missing values into a complete dataset.

## Key Features

- Offers full implementation of **structured** and unstructured missingness mechanisms (MCAR, MAR, MNAR).
- Extends existing missingness simulations along 3 additional dimensions: 
    1. **Structured** and unstructured missingness
    2. Probabilistic/deterministic missingness
    3. Sequential/block missingness.
- Flexibility and control in distribution and parameters for coefficient generation in logistical model, correlation of block missingness between individual features
- `seed` parameter ensures full reproducibility of randomly generation variables

<!-- ## Implementation Details

Implementation is provided for each of the following mechanisms:
1. MCAR
    - MCAR Unstructured
    - MCAR Weak, Block
    - MCAR Weak, Sequential
    - MCAR Strong, Block
    - MCAR Strong, Sequential
2. MAR
    - MAR Unstructured, Probabilistic
    - MAR Unstructured, Deterministic
    - MAR Weak, Block
    - MAR Weak, Sequential
    - MAR Strong, Block
    - MAR Strong, Sequential
3. MNAR
    - MNAR Unstructured, Probabilistic
    - MNAR Unstructured, Deterministic
    - MNAR Weak, Block
    - MNAR Weak, Sequential
    - MNAR Strong, Block
    - MNAR Strong, Sequential -->

## Functions

`simulate_nan(X, p_miss, mecha, ...)`

This is the main entry point for the simulation. 

**Parameters**: 
- X : np.ndarray
    - Input dataset (complete data, no missing values).
- p_miss : float
    - Proportion of missing values to introduce.
- mecha : str, default="MCAR"
    - Missing data mechanism. One of:
        - "MCAR" : Missing Completely At Random
        - "MAR"  : Missing At Random
        - "MNAR" : Missing Not At Random
- structured : bool, default=False
    - If True, generates structured missingness 
- weak : bool, default=True
    - If True, weak/probabilistic missingness. If False, strong/deterministic missingness.
- sequential : bool, default=False
    - If True, sequential missingness. If False, block missingness. 
- p_obs : float, optional
    - Proportion of observed entries (used in MAR/MNAR missingness).
- exclude_inputs : bool, default=True
    - Used in MNAR missingness. If True, mechanism (ii) used. If False, mechanism (i) used. 
- cov : np.ndarray, optional
    - Covariance matrix used to generate latent variables for block missingness. Randomly generated if unspecified. 
- coeff_dist : str, default="uniform"
    - Distribution to draw coefficients from. One of:
        - "uniform" : Uniform distribution
        - "normal" : Normal distribution
        - "betas" : Manually-specified coefficients
- coeff_arg0 : float, default=0.0
    - First parameter of coefficient distribution. If coeff_dist is:
        - "uniform" : minimum value
        - "normal" : distribution mean
        - "betas" : list of coefficients
- coeff_arg1 : float, default=1.0
    - Second parameter of coefficient distribution. If coeff_dist is:
        - "uniform" : maximum value
        - "normal" : distribution standard deviation
        - "betas" : unused
- idxs_obs : List[int], optional
    - List of feature indexes of observed variables (used for MAR/MNAR). Randomly generated if unspecified. 
- seed : int, default=None
    - Random seed for reproducibility.

**Returns**: 
Dictionary containing:
- 'X_init' : np.ndarray
    - Original dataset (intact, float64).
- 'X_incomp' : np.ndarray
    - Dataset with missing values (NaN-s inserted).
- 'mask' : np.ndarray
    - Boolean mask of missing values (True if missing).
- 'cov' : np.ndarray
    - Covariance matrix used (or None if not applicable).
- 'seed' : int
    - Random seed used.

## Parameters to Generate each Missingness Mechanism

| Mechanism | `mecha` | `structured` | `weak` | `sequential` |
| ------- | ------- | ------- | ------- | ------- |
| MCAR Unstructured | 'MCAR' |  `False` |  - |  - |
| MCAR Weak, Block | 'MCAR' |  `True` |  `True` |  `False` |
| MCAR Weak, Sequential | 'MCAR' |  `True` |  `True` |  `True` |
| MCAR Strong, Sequential | 'MCAR' |  `True` |  `False` |  `False` |
| MCAR Strong, Sequential | 'MCAR' |  `True` |  `False` |  `True` |
| MAR Unstructured, Probabilistic | 'MAR' |  `False` |  `True` |  - |
| MAR Unstructured, Deterministic | 'MAR' |  `False` |  `False` |  - |
| MAR Weak, Block | 'MAR' |  `True` |  `True` |  `False` |
| MAR Weak, Sequential | 'MAR' |  `True` |  `True` |  `True` |
| MAR Strong, Sequential | 'MAR' |  `True` |  `False` |  `False` |
| MAR Strong, Sequential | 'MAR' |  `True` |  `False` |  `True` |
| MNAR Unstructured, Probabilistic | 'MNAR' |  `False` |  `True` |  - |
| MNAR Unstructured, Deterministic | 'MNAR' |  `False` |  `False` |  - |
| MNAR Weak, Block | 'MNAR' |  `True` |  `True` |  `False` |
| MNAR Weak, Sequential | 'MNAR' |  `True` |  `True` |  `True` |
| MNAR Strong, Sequential | 'MNAR' |  `True` |  `False` |  `False` |
| MNAR Strong, Sequential | 'MNAR' |  `True` |  `False` |  `True` |

## Example Usage

```python
import numpy as np
from simulate import simulate_nan

# Generate simulation data
X_complete_cont = np.random.rand(1000, 4)

# 40% missingness, MCAR-WS Sequential
X_miss_mcar_WSS = simulate_nan(X_complete_cont, p_miss=0.4, mecha="MCAR", structured="True", weak="True", sequential="True") 

# 20% missingness, MAR-UD
X_miss_mar_UD = simulate_nan(X_complete_cont, p_miss=0.2, mecha="MAR", structured="False", weak="False") 

# 20% missingness, MAR-WS Sequential, coefficients drawn from uniform distribution [0,2)
X_miss_mar_WSS = simulate_nan(X_complete_cont, p_miss=0.2, mecha="MAR", structured="True", weak="True", sequential="True", coeff_dist="uniform", coeff_arg0=0, coeff_arg1=2) 

# 50% missingness, MNAR-SS Block, manually-specified observed columns
X_miss_mnar_SSB = simulate_nan(X_complete_cont, p_miss=0.5, mecha="MNAR", structured="True", weak="False", sequential="False", idxs_obs=[1, 3]) 

# 40% missingness, MCAR-WS Block, manually-specified covariance matrix
cov = np.array([[1,0.4,-0.8,0.05],[0.4,1,0.2,-0.5],[-0.8,0.2,1,0.3],[0.05,-0.5,0.3,1]])
X_miss_mcar_WSB = simulate_nan(X_complete_cont, p_miss=0.4, mecha="MCAR", structured="True", weak="True", sequential="False", cov=cov) 


```

## Internal Functions

The module is built on several internal functions that handle the specifics of each missingness mechanism.

`MCAR_mask(...), MAR_mask(...), MNAR_mask_logistic(...)`
- These functions generate the missingness mask for each specific mechanism (MCAR, MAR, MNAR). They are called internally by simulate_nan.

`pick_coeffs(...)`
- This utility function selects and normalizes the coefficients for the logistic model based on the chosen distribution and parameters.

`fit_intercepts(...)`
- This function finds the intercept value for the logistic model using a bisection method, ensuring the target proportion of missing values ($p_{miss}$) is achieved.

## References
<a id="1">[1]</a> 
Rubin, Donald B. “Inference and Missing Data.” Biometrika, vol. 63, no. 3, 1976, pp. 581–92. DOI.org (Crossref), https://doi.org/10.1093/biomet/63.3.581.

<a id="2">[2]</a> 
Jackson, James, et al. “A Complete Characterisation of Structured Missingness.” arXiv:2307.02650, arXiv, 5 Jul. 2023. arXiv.org, https://doi.org/10.48550/arXiv.2307.02650.

