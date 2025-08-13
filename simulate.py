# stdlib
from typing import List

# third party
import numpy as np
from scipy import optimize
from scipy.special import expit

# utils
# from sim_utils import normalize
# from sim_utils import generate_coeffs
# from sim_utils import round_near_zero
# from sim_utils import generate_random_cov

def normalize(data, coeffs, tol=1e-10):
    std = np.std(data@coeffs, axis=0, keepdims=True)
    std[np.isclose(std, 0.0, atol=tol)] = 1.0 # TODO: Quantify effects of doing 
    return coeffs / std

def generate_random_cov(n):
    A = np.random.randn(n, n)
    S = A @ A.T
    diag_sqrt = np.sqrt(np.diag(S))
    norm_matrix = np.outer(diag_sqrt, diag_sqrt)
    correlation_matrix = S / norm_matrix
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix

def round_near_zero(arr: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    if (arr < 0).all():
        arr[(arr) > -tolerance] = 0.0
    if (arr > 0).all():
        arr[(arr) < tolerance] = -tolerance
    return arr

def generate_coeffs(shape,
                    dist: str="uniform",
                    arg0: float=0.0,
                    arg1: float=1.0):
    if dist == "uniform":
        min = arg0
        max = arg1
        if min >= max:
            raise ValueError("Minimum and maximum endpoints mismatched.")
        if type(shape) == tuple:
            return np.random.rand(*shape) * (max-min) + min
        else:
            return np.random.rand(shape) * (max- min) + min
    elif dist == "normal":
        mean = arg0
        std = arg1
        return np.random.normal(loc=mean, scale=std, size=shape)
    elif dist == "betas":
        beta = next(arg0) if hasattr(arg0, '__next__') else arg0
        return np.full(shape, beta)
    else:
        raise TypeError("Unrecognized coefficient distribution.")

def pick_coeffs(
    X: np.ndarray,
    idxs_obs: List[int] = [],
    idxs_nas: List[int] = [],
    self_mask: bool = False,
    latent_component: np.ndarray = None,
    mask_component: np.ndarray = None,
    coeff_dist: str = "uniform",
    coeff_arg0: float = 0.0,
    coeff_arg1: float = 1.0
    ) -> np.ndarray:

    if self_mask: # MNAR specific implementation
        pass
    else:
        d_na = len(idxs_nas)
        if X is not None:
            n, d = X.shape
            d_obs = len(idxs_obs) 
            inputs = X[:, idxs_obs]
            coeffs = generate_coeffs((d_obs, d_na), coeff_dist, coeff_arg0, coeff_arg1)          
            coeffs = normalize(inputs, coeffs)
            inputs_and_coeffs = np.dot(inputs, coeffs)
        if latent_component is not None:
            if X is not None:
                inputs_and_coeffs += latent_component 
            else:
                inputs_and_coeffs = latent_component 
        elif mask_component is not None:
            mask_coeffs = generate_coeffs((mask_component.shape[1], d_na), coeff_dist, coeff_arg0, coeff_arg1)
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
              coeff_dist: str = "uniform",
              coeff_arg0: float = 0.0,
              coeff_arg1: float = 1.0) -> np.ndarray:
    if structured == False:
        # MCAR Unstructured  
        mask = np.random.rand(*X.shape) < p_miss # (I)
    else:       
        n, d = X.shape
        mask = np.zeros((n, d)).astype(bool)
        
        if weak and not sequential:
            # MCAR Weak + Block 
            if cov is None:
                cov = generate_random_cov(d)
            mean = np.zeros(d)
            latent_effects = np.random.multivariate_normal(mean, cov, size=n)
            for j in np.arange(d):
                inputs_and_coeffs = pick_coeffs(X=None, idxs_obs = None, idxs_nas=[j], latent_component=latent_effects[:,j].reshape(-1, 1), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n,1)
                mask[:, j] = (ber < ps).flatten()
        
        elif weak and sequential: 
            # MCAR Weak + Sequential 
            for j in np.arange(d):
                inputs_and_coeffs = pick_coeffs(X=None, idxs_obs=None, idxs_nas=[j], mask_component=mask[:, :j], self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)  # how does this handle the first index? (it does MCAR!)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()
        
        elif not weak and not sequential:
            # MCAR Strong + Block 
            if cov is None:
                cov = generate_random_cov(d)
            mean = np.zeros(d)
            latent_effects = np.random.multivariate_normal(mean, cov, size=n)
            for j in np.arange(d):
                inputs_and_coeffs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], latent_component=latent_effects[:,j].reshape(-1, 1), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
        
        else: 
            # MCAR Strong + Sequential 
            for j in np.arange(d): # TODO: spread out the missingness!
                if j == 0:
                    mask[:, j] = np.random.rand(mask.shape[0]) < p_miss
                else:              
                    inputs_and_coeffs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], mask_component=(mask[:, :j]), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)   
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
              coeff_dist: str = "uniform",
              coeff_arg0: float = 0.0,
              coeff_arg1: float = 1.0,
              idxs_obs: np.ndarray = None) -> np.ndarray:
    #TODO: CHANGE THE SIZE OF LATENTS HERE!!
    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1)  # number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  # number of variables that will have missing values

    if idxs_obs is None:
        idxs_obs = np.random.choice(d, d_obs, replace=False) # randomly generate indexes of variables with have no missing values

    idxs_nas = np.setdiff1d(np.arange(d), idxs_obs) # indexes of variables that will have missing values

    if structured == False:
        if weak: # MAR Probabilistic (VI)
            # MAR Probabilistic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()
        else:
            # MAR Deterministic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
    else:

        if weak and not sequential:
            # MAR Weak + Block 
            if cov is None:
                cov = generate_random_cov(d_na)
            mean = np.zeros(d_na)
            latent_effects = np.random.multivariate_normal(mean, cov, size=n)
            for col_idx, j in enumerate(idxs_nas):
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, latent_component=latent_effects[:, col_idx].reshape(-1, 1), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif weak and sequential:
            # MAR Weak + Sequential 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, mask_component=mask[:, :j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif not weak and not sequential:
            # MAR Strong + Block
            if cov is None:
                cov = generate_random_cov(d_na)
            mean = np.zeros(d_na)
            latent_effects = np.random.multivariate_normal(mean, cov, size=n)
            for col_idx, j in enumerate(idxs_nas):
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, latent_component=(latent_effects[:, col_idx].reshape(-1, 1)), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
            
        else:
            # MAR Strong + Sequential
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, mask_component=(mask[:, :j]), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
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
    coeff_dist: str = "uniform",
    coeff_arg0: float = 0.0,
    coeff_arg1: float = 1.0,
    idxs_obs: np.ndarray = None
    ) -> np.ndarray:

    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of inputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Args:
        X : Data for which missing values will be simulated.
        p : Proportion of missing values to generate for variables which will have missing values.
        p_params : Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
        exclude_inputs : True: mechanism (ii) is used, False: (i)

    Returns:
        mask : Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) if exclude_inputs else d
    d_na = (d - d_obs if exclude_inputs else d)

    if idxs_obs is None:
        idxs_obs = (np.random.choice(d, d_obs, replace=False)) if exclude_inputs else np.arange(d) # randomly generate indexes of variables with have no missing values
    idxs_nas = (np.setdiff1d(np.arange(d), idxs_obs)) if exclude_inputs else np.arange(d) # indexes of variables that will have missing values

    if structured == False:
        if weak:
            # MNAR Probabilistic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()
        else: 
            # MNAR Deterministic 
            for j in idxs_nas:
                inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
        
        if exclude_inputs:
            mask[:, idxs_obs] = np.random.rand(n, d_obs) < p_miss
    else:

        if weak and not sequential:
            # MNAR Weak + Block 
            if cov is None:
                cov = generate_random_cov(d)
            mean = np.zeros(d)
            latent_effects = np.random.multivariate_normal(mean, cov, size=n)
            for j in np.arange(d): 
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, latent_component=latent_effects[:,j].reshape(-1, 1), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                else: # M_obs terms in exclude_inputs = True case
                    inputs_and_coeffs = pick_coeffs(None, idxs_obs=[], idxs_nas=[j], latent_component=latent_effects[:,j].reshape(-1, 1), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1) 
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif weak and sequential: 
            # MNAR Weak + Sequential 
            for j in np.arange(d): 
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, mask_component=mask[:, :j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                else:
                    inputs_and_coeffs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], self_mask=False, mask_component=mask[:, :j], coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = expit(inputs_and_coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif not weak and not sequential: 
            # MNAR Strong + Block 
            if cov is None:
                cov = generate_random_cov(d)
            mean = np.zeros(d)
            latent_effects = np.random.multivariate_normal(mean, cov, size=n)
            for j in np.arange(d): 
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, latent_component=(latent_effects[:, j].reshape(-1, 1)), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                else: #M_obs terms in exclude_inputs = True case
                    inputs_and_coeffs = pick_coeffs(None, idxs_obs=[], idxs_nas=[j], latent_component=(latent_effects[:, j].reshape(-1, 1)), self_mask=False, coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
            
        else:
            # MNAR Strong + Sequential 
            for j in np.arange(d): 
                if j == 0 and j not in idxs_nas:
                    mask[:, j] = np.random.rand(mask.shape[0]) < p_miss
                else:
                    if j in idxs_nas: # this is always true for exclude_inputs=False
                        inputs_and_coeffs = pick_coeffs(X, idxs_obs, [j], self_mask=False, mask_component=(mask[:, :j]), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                    else: # M_obs terms
                        inputs_and_coeffs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], self_mask=False, mask_component=(mask[:, :j]), coeff_dist=coeff_dist, coeff_arg0=coeff_arg0, coeff_arg1=coeff_arg1)
                    intercepts = fit_intercepts(inputs_and_coeffs, p_miss, weak)
                    ps = round_near_zero(inputs_and_coeffs + intercepts) >= 0
                    mask[:, j] = ps.flatten()

    return mask, cov

def simulate_nan(X: np.ndarray,
                 p_miss: float,
                 mecha: str = "MCAR",
                 opt: str = "logistic",
                 structured: bool=False,
                 weak: bool=True,
                 sequential: bool=False,
                 p_obs: float=0.5,
                 exclude_inputs: bool=True,
                 cov: float=None,
                 coeff_dist: str = "normal",
                 coeff_arg0: float = 0.0,
                 coeff_arg1: float = 1.0,
                 idxs_obs: np.ndarray = None
                 ) -> np.ndarray:
    if mecha == "MAR":
        mask, cov = MAR_mask(X, p_miss, structured, weak, sequential, p_obs, cov, coeff_dist, coeff_arg0, coeff_arg1, idxs_obs)
    elif mecha == "MNAR" and opt == "logistic":
        mask, cov = MNAR_mask_logistic(X, p_miss, structured, weak, sequential, p_obs, exclude_inputs, cov, coeff_dist, coeff_arg0, coeff_arg1, idxs_obs)
    else:
        mask, cov = MCAR_mask(X, p_miss, structured, weak, sequential, cov, coeff_dist, coeff_arg0, coeff_arg1)
    X_nas = X.copy()
    X_nas[mask.astype(bool)] = np.nan
    return {'X_init': X.astype(np.float64), 'X_incomp': X_nas.astype(np.float64), 'mask': mask, 'cov': cov}
