# stdlib
from typing import List

# third party
import numpy as np
from scipy import optimize
from scipy.special import expit

# utils
from sim_utils import normalize
from sim_utils import generate_coeffs
from sim_utils import round_near_zero
from sim_utils import generate_random_cov

def pick_coeffs(
    X: np.ndarray,
    rng: np.random.Generator,
    idxs_obs: List[int] = [],
    idxs_nas: List[int] = [],
    self_mask: bool = False,
    latent_component: np.ndarray = None,
    mask_component: np.ndarray = None,
    coeff_dist: str = "uniform",
    coeff_arg0: float = 0.0,
    coeff_arg1: float = 1.0,
    ) -> np.ndarray:

    if self_mask: # MNAR specific implementation
        pass
    else:
        d_na = len(idxs_nas)
        if X is not None:
            n, d = X.shape
            d_obs = len(idxs_obs) 
            inputs = X[:, idxs_obs]
            coeffs = generate_coeffs((d_obs, d_na), rng, coeff_dist, coeff_arg0, coeff_arg1)          
            coeffs = normalize(inputs, coeffs)
            inputs_and_coeffs = np.dot(inputs, coeffs)
        if latent_component is not None:
            if X is not None:
                inputs_and_coeffs += latent_component 
            else:
                inputs_and_coeffs = latent_component 
        elif mask_component is not None:
            mask_coeffs = generate_coeffs((mask_component.shape[1], d_na), rng, coeff_dist, coeff_arg0, coeff_arg1)
            if X is not None:
                inputs_and_coeffs += np.dot(mask_component, mask_coeffs)
            else:
                inputs_and_coeffs = np.dot(mask_component, mask_coeffs)
    return inputs_and_coeffs

def fit_intercepts(
        inputs_and_coeffs: np.ndarray,
        p_miss: float,
        weak: bool = True,
        self_mask: bool = False
        ) -> np.ndarray:
    
    if self_mask: # MNAR specific implementation
        pass
    else:
        if weak: # probabilistic 
            def f(x: np.ndarray) -> np.ndarray:
                return expit(inputs_and_coeffs+x).mean().item() - p_miss
            intercepts = optimize.bisect(f, -150, 150) 
        else: # deterministic
            def f(x: np.ndarray) -> np.ndarray:
                return ((inputs_and_coeffs+x) >= 0).mean().item() - p_miss
            intercepts = optimize.bisect(f, -150, 150) 
    return intercepts

def MCAR_mask(X: np.ndarray,
              p_miss: float,
              structured: bool=False,
              weak: bool=True,
              sequential: bool=False,
              cov: float=None,
              effect_strength: float=None,
              coeff_dist: str = "uniform",
              coeff_arg0: float = 0.0,
              coeff_arg1: float = 1.0,
              rng: np.random.Generator=None) -> np.ndarray:
    """
    Missing Completely At Random (MCAR) mechanisms with a logistic masking model. 
    Missingness is generated for all features using logistic model.

    Parameters
    ----------
    X : np.ndarray
        Input dataset (complete data, no missing values).
    p_miss : float
        Proportion of missing values to introduce.
    structured : bool, default=False
        If True, structured MCAR. If False, unstructured MCAR.
    weak : bool, default=True
        If True, weak/probabilistic MCAR. If False, strong MCAR.
    sequential : bool, default=False
        If True, sequential MCAR. If False, block MCAR. 
    cov : np.ndarray, optional
        Covariance matrix used to generate latent variables for block missingness. Randomly generated if unspecified. 
    coeff_dist : str, default="normal"
        Distribution to draw coefficients from. One of:
            - "uniform" : Uniform distribution
            - "normal" : Normal distribution
            - "betas" : Manually-specified coefficients
    coeff_arg0 : float, default=0.0
        First parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : minimum value
            - "normal" : distribution mean
            - "betas" : list of coefficients
    coeff_arg1 : float, default=1.0
        Second parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : maximum value
            - "normal" : distribution standard deviation
            - "betas" : unused
    idxs_obs : List[int], optional
        List of feature indexes of observed variables. Randomly generated if unspecified. 
    seed : int, default=None
        Random seed for reproducibility.

    Returns
    -------
    - mask : np.ndarray
        Boolean mask of missing values (True if missing).
    - cov : np.ndarray
        Covariance matrix used (or None if not applicable).

    """
    if rng is None:
        rng = np.random.default_rng()
    
    if structured == False:
        # MCAR Unstructured  
        mask = rng.random((X.shape)) < p_miss # (I)
    else:       
        n, d = X.shape
        mask = np.zeros((n, d)).astype(bool)
        
        if weak and not sequential:
            # MCAR Weak + Block 
            if cov is None:
                cov = generate_random_cov(d, rng)
            if effect_strength is None:
                effect_strength = rng.uniform(1, 2)
            mean = np.zeros(d)
            latent_effects = rng.multivariate_normal(mean, cov, size=n)*effect_strength
            for j in np.arange(d):
                inputs_and_coeffs = pick_coeffs(X=None, rng=rng, idxs_obs = None, idxs_nas=[j], latent_component=latent_effects[:,j].reshape(-1, 1), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n,1))
                mask[:, j] = (ber < ps).flatten()
        
        elif weak and sequential: 
            # MCAR Weak + Sequential 
            for j in np.arange(d):
                inputs_and_coeffs = pick_coeffs(X=None, rng=rng, idxs_obs=None, idxs_nas=[j], mask_component=mask[:, :j], self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)  # how does this handle the first index? (it does MCAR!)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n, 1))
                mask[:, j] = (ber < ps).flatten()
        
        elif not weak and not sequential:
            # MCAR Strong + Block 
            if cov is None:
                cov = generate_random_cov(d, rng)
            if effect_strength is None:
                effect_strength = rng.uniform(1, 2)
            mean = np.zeros(d)
            latent_effects = rng.multivariate_normal(mean, cov, size=n)*effect_strength
            for j in np.arange(d):
                inputs_and_coeffs = pick_coeffs(X=None, rng=rng, idxs_obs=[], idxs_nas=[j], latent_component=latent_effects[:,j].reshape(-1, 1), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
        
        else: 
            # MCAR Strong + Sequential 
            for j in np.arange(d): # TODO: spread out the missingness!
                if j == 0:
                    mask[:, j] = rng.random((mask.shape[0])) < p_miss
                else:              
                    inputs_and_coeffs = pick_coeffs(X=None, rng=rng, idxs_obs=[], idxs_nas=[j], mask_component=(mask[:, :j]), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)   
                    intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                    ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                    mask[:, j] = ps.flatten()
    return mask.astype(float), cov
 
def MAR_mask(X: np.ndarray,
              p_miss: float,
              structured: bool=False,
              weak: bool=True,
              sequential: bool=False,
              p_obs: float=0.5,
              cov: float=None,
              effect_strength: float=None,
              coeff_dist: str = "uniform",
              coeff_arg0: float = 0.0,
              coeff_arg1: float = 1.0,
              idxs_obs: np.ndarray = None,
              rng: np.random.Generator=None) -> np.ndarray:
    """
    Missing At Random (MAR) mechanisms with a logistic masking model. 
    Features are split into a set of missing and observed values, missingness is generated for missing features using logistic model.

    Parameters
    ----------
    X : np.ndarray
        Input dataset (complete data, no missing values).
    p_miss : float
        Proportion of missing values to introduce.
    structured : bool, default=False
        If True, structured MAR. If False, unstructured MAR.
    weak : bool, default=True
        If True, weak/probabilistic MAR. If False, strong/deterministic MAR.
    sequential : bool, default=False
        If True, sequential MAR. If False, block MAR. 
    p_obs : float, optional
        Proportion of observed entries. Used if exclude_inputs is true. 
    cov : np.ndarray, optional
        Covariance matrix used to generate latent variables for block missingness. Randomly generated if unspecified. 
    coeff_dist : str, default="normal"
        Distribution to draw coefficients from. One of:
            - "uniform" : Uniform distribution
            - "normal" : Normal distribution
            - "betas" : Manually-specified coefficients
    coeff_arg0 : float, default=0.0
        First parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : minimum value
            - "normal" : distribution mean
            - "betas" : list of coefficients
    coeff_arg1 : float, default=1.0
        Second parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : maximum value
            - "normal" : distribution standard deviation
            - "betas" : unused
    idxs_obs : List[int], optional
        List of feature indexes of observed variables. Randomly generated if unspecified. 
    seed : int, default=None
        Random seed for reproducibility.

    Returns
    -------
    - mask : np.ndarray
        Boolean mask of missing values (True if missing).
    - cov : np.ndarray
        Covariance matrix used (or None if not applicable).

    """

    if rng is None:
        rng = np.random.default_rng()
    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1)  # number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  # number of variables that will have missing values

    if idxs_obs is None:
        idxs_obs = rng.choice(d, d_obs, replace=False) # randomly generate indexes of variables with have no missing values

    idxs_nas = np.setdiff1d(np.arange(d), idxs_obs) # indexes of variables that will have missing values

    if structured == False:
        if weak: # MAR Probabilistic (VI)
            # MAR Probabilistic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n, 1))
                mask[:, j] = (ber < ps).flatten()
        else:
            # MAR Deterministic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
    else:

        if weak and not sequential:
            # MAR Weak + Block 
            if cov is None:
                cov = generate_random_cov(d_na, rng)
            if effect_strength is None:
                effect_strength = rng.uniform(1, 2)
            mean = np.zeros(d_na)
            latent_effects = rng.multivariate_normal(mean, cov, size=n)*effect_strength
            for i, j in enumerate(idxs_nas):
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], self_mask=False, latent_component=latent_effects[:, i].reshape(-1, 1), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n, 1))
                mask[:, j] = (ber < ps).flatten()

        elif weak and sequential:
            # MAR Weak + Sequential 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], self_mask=False, mask_component=mask[:, :j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n, 1))
                mask[:, j] = (ber < ps).flatten()

        elif not weak and not sequential:
            # MAR Strong + Block
            if cov is None:
                cov = generate_random_cov(d_na, rng)
            if effect_strength is None:
                effect_strength = rng.uniform(1, 2)
            mean = np.zeros(d_na)
            latent_effects = rng.multivariate_normal(mean, cov, size=n)*effect_strength
            for i, j in enumerate(idxs_nas):
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], self_mask=False, latent_component=(latent_effects[:, i].reshape(-1, 1)), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
            
        else:
            # MAR Strong + Sequential
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], self_mask=False, mask_component=(mask[:, :j]), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
    return mask.astype(float), cov

def MNAR_mask_logistic(
    X: np.ndarray,
    p_miss: float,
    structured: bool = False,
    weak: bool = True,
    sequential: bool = False,
    p_obs: float = 0.3,
    exclude_inputs: bool = True,
    cov: float=None,
    effect_strength: float=None,
    coeff_dist: str = "uniform",
    coeff_arg0: float = 0.0,
    coeff_arg1: float = 1.0,
    idxs_obs: np.ndarray = None,
    rng: np.random.Generator=None
    ) -> np.ndarray:

    """
    Missing Not At Random (MNAR) mechanisms with a logistic masking model. It implements two mechanisms:
    (i) Missingness is generated taking all features as inputs. 
    (ii) Features are split into a set of missing and observed values for a logistic model. Missing values masked MAR
    using observed values logistic model. Observed values masked MCAR. 

    Parameters
    ----------
    X : np.ndarray
        Input dataset (complete data, no missing values).
    p_miss : float
        Proportion of missing values to introduce.
    structured : bool, default=False
        If True, structured MNAR. If False, unstructured MNAR.
    weak : bool, default=True
        If True, weak/probabilistic MNAR. If False, strong/deterministic MNAR.
    sequential : bool, default=False
        If True, sequential MNAR. If False, block MNAR. 
    p_obs : float, optional
        Proportion of observed entries. Used if exclude_inputs is true. 
    exclude_inputs : bool, default=True
        If True, mechanism (ii) used. If False, mechanism (i) used. 
    cov : np.ndarray, optional
        Covariance matrix used to generate latent variables for block missingness. Randomly generated if unspecified. 
    coeff_dist : str, default="normal"
        Distribution to draw coefficients from. One of:
            - "uniform" : Uniform distribution
            - "normal" : Normal distribution
            - "betas" : Manually-specified coefficients
    coeff_arg0 : float, default=0.0
        First parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : minimum value
            - "normal" : distribution mean
            - "betas" : list of coefficients
    coeff_arg1 : float, default=1.0
        Second parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : maximum value
            - "normal" : distribution standard deviation
            - "betas" : unused
    idxs_obs : List[int], optional
        List of feature indexes of observed variables. Randomly generated if unspecified. 
    seed : int, default=None
        Random seed for reproducibility.

    Returns
    -------
    - mask : np.ndarray
        Boolean mask of missing values (True if missing).
    - cov : np.ndarray
        Covariance matrix used (or None if not applicable).

    """

    if rng is None:
        rng = np.random.default_rng()

    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) if exclude_inputs else d
    d_na = (d - d_obs if exclude_inputs else d)

    if idxs_obs is None:
        idxs_obs = (rng.choice(d, d_obs, replace=False)) if exclude_inputs else np.arange(d) # randomly generate indexes of variables with have no missing values
    idxs_nas = (np.setdiff1d(np.arange(d), idxs_obs)) if exclude_inputs else np.arange(d) # indexes of variables that will have missing values

    if structured == False:
        if weak:
            # MNAR Probabilistic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n, 1))
                mask[:, j] = (ber < ps).flatten()
        else: 
            # MNAR Deterministic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, rng, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
        
        if exclude_inputs:
            mask[:, idxs_obs] = rng.random((n, len(idxs_obs))) < p_miss
    else:

        if weak and not sequential:
            # MNAR Weak + Block 
            if cov is None:
                cov = generate_random_cov(d, rng)
            if effect_strength is None:
                effect_strength = rng.uniform(1, 2)
            mean = np.zeros(d)
            latent_effects = rng.multivariate_normal(mean, cov, size=n)*effect_strength
            for j in np.arange(d): 
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    inputs_and_coeffs = pick_coeffs(X,rng, idxs_obs, [j], self_mask=False, latent_component=latent_effects[:,j].reshape(-1, 1), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                else: # M_obs terms in exclude_inputs = True case
                    inputs_and_coeffs = pick_coeffs(X=None, rng=rng,idxs_obs=[], idxs_nas=[j], latent_component=latent_effects[:,j].reshape(-1, 1), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1) 
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n, 1))
                mask[:, j] = (ber < ps).flatten()

        elif weak and sequential: 
            # MNAR Weak + Sequential 
            for j in np.arange(d): 
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    inputs_and_coeffs = pick_coeffs(X, rng,idxs_obs, [j], self_mask=False, mask_component=mask[:, :j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                else:
                    inputs_and_coeffs = pick_coeffs(X=None, rng=rng,idxs_obs=[], idxs_nas=[j], self_mask=False, mask_component=mask[:, :j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = rng.random((n, 1))
                mask[:, j] = (ber < ps).flatten()

        elif not weak and not sequential: 
            # MNAR Strong + Block 
            if cov is None:
                cov = generate_random_cov(d, rng)
            if effect_strength is None:
                effect_strength = rng.uniform(1, 2)
            mean = np.zeros(d)
            latent_effects = rng.multivariate_normal(mean, cov, size=n)*effect_strength
            for j in np.arange(d): 
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    inputs_and_coeffs = pick_coeffs(X, rng,idxs_obs, [j], self_mask=False, latent_component=(latent_effects[:, j].reshape(-1, 1)), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                else: #M_obs terms in exclude_inputs = True case
                    inputs_and_coeffs = pick_coeffs(None, rng,idxs_obs=[], idxs_nas=[j], latent_component=(latent_effects[:, j].reshape(-1, 1)), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
            
        else:
            # MNAR Strong + Sequential 
            for j in np.arange(d): 
                if j == 0 and j not in idxs_nas:
                    mask[:, j] = rng.random((mask.shape[0])) < p_miss
                else:
                    if j in idxs_nas: # this is always true for exclude_inputs=False
                        inputs_and_coeffs = pick_coeffs(X, rng,idxs_obs, [j], self_mask=False, mask_component=(mask[:, :j]), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                    else: # M_obs terms
                        inputs_and_coeffs = pick_coeffs(X=None, rng=rng,idxs_obs=[], idxs_nas=[j], self_mask=False, mask_component=(mask[:, :j]), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                    intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                    ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                    mask[:, j] = ps.flatten()
    return mask, cov

def simulate_nan(X: np.ndarray,
                 p_miss: float,
                 mecha: str = "MCAR",
                 structured: bool=False,
                 weak: bool=True,
                 sequential: bool=False,
                 p_obs: float=0.5,
                 exclude_inputs: bool=True,
                 cov: float=None,
                 effect_strength: float=None,
                 coeff_dist: str = "normal",
                 coeff_arg0: float = 0.0,
                 coeff_arg1: float = 1.0,
                 idxs_obs: np.ndarray = None,
                 seed: int = None
                 ) -> np.ndarray:
    """
    Generate missingness mask for specified dataset at specified proportion using structured and unstructured missingnes mechanisms. 
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
        - MNAR Strong, Sequential

    Missing Not At Random (MNAR) mechanisms is implemented with two mechanisms:
    (i) Missingness is generated taking all features as inputs. 
    (ii) Features are split into a set of missing and observed values for a logistic model. Missing values masked MAR
    using observed values logistic model. Observed values masked MCAR. 

    Parameters
    ----------
    X : np.ndarray
        Input dataset (complete data, no missing values).
    p_miss : float
        Proportion of missing values to introduce.
    mecha : str, default="MCAR"
        Missing data mechanism. One of:
            - "MCAR" : Missing Completely At Random
            - "MAR"  : Missing At Random
            - "MNAR" : Missing Not At Random
    structured : bool, default=False
        If True, generates structured missingness 
    weak : bool, default=True
        If True, weak/probabilistic missingness. If False, strong/deterministic missingness.
    sequential : bool, default=False
        If True, sequential missingness. If False, block missingness. 
    p_obs : float, optional
        Proportion of observed entries (used in MAR/MNAR missingness).
    exclude_inputs : bool, default=True
        Used in MNAR missingness. If True, mechanism (ii) used. If False, mechanism (i) used. 
    cov : np.ndarray, optional
        Covariance matrix used to generate latent variables for block missingness. Randomly generated if unspecified. 
    coeff_dist : str, default="normal"
        Distribution to draw coefficients from. One of:
            - "uniform" : Uniform distribution
            - "normal" : Normal distribution
            - "betas" : Manually-specified coefficients
    coeff_arg0 : float, default=0.0
        First parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : minimum value
            - "normal" : distribution mean
            - "betas" : list of coefficients
    coeff_arg1 : float, default=1.0
        Second parameter of coefficient distribution.
        If coeff_dist is:
            - "uniform" : maximum value
            - "normal" : distribution standard deviation
            - "betas" : unused
    idxs_obs : List[int], optional
        List of feature indexes of observed variables (used for MAR/MNAR). Randomly generated if unspecified. 
    seed : int, default=None
        Random seed for reproducibility.

    Returns
    -------
    Dictionary containing:
    - 'X_init' : np.ndarray
        Original dataset (intact, float64).
    - 'X_incomp' : np.ndarray
        Dataset with missing values (NaN-s inserted).
    - 'mask' : np.ndarray
        Boolean mask of missing values (True if missing).
    - 'cov' : np.ndarray
        Covariance matrix used (or None if not applicable).
    - 'seed' : int
        Random seed used.
    """
    
    rng = np.random.default_rng(seed)

    if mecha == "MAR":
        mask, cov = MAR_mask(X, p_miss, structured, weak, sequential, p_obs, cov, effect_strength, coeff_dist, coeff_arg0, coeff_arg1, idxs_obs,rng=rng)
    elif mecha == "MNAR":
        mask, cov = MNAR_mask_logistic(X, p_miss, structured, weak, sequential, p_obs, exclude_inputs, cov, effect_strength, coeff_dist, coeff_arg0, coeff_arg1, idxs_obs,rng=rng)
    else:
        mask, cov = MCAR_mask(X, p_miss, structured, weak, sequential, cov, effect_strength, coeff_dist, coeff_arg0, coeff_arg1,rng=rng)

    X_nas = X.copy()
    X_nas[mask.astype(bool)] = np.nan
    return {'X_init': X.astype(np.float64), 'X_incomp': X_nas.astype(np.float64), 'mask': mask, 'cov': cov, 'seed': seed}

