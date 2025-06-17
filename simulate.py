# stdlib
from typing import List

# third party
import numpy as np
from scipy import optimize
from scipy.special import expit

# utils
from sim_utils import block_assignment
from sim_utils import normalize
from sim_utils import random_flip
from sim_utils import round_near_zero
from sim_utils import random_negate

def pick_coeffs(
    X: np.ndarray,
    idxs_obs: List[int] = [],
    idxs_nas: List[int] = [],
    self_mask: bool = False,
    struc_component: np.ndarray = None
    ) -> np.ndarray:

    if self_mask: # MNAR specific implementation
        if X is not None:
            n, d = X.shape
            inputs = X
            coeffs = np.random.rand(d)
            Wx = inputs * coeffs
            coeffs /= np.std(Wx, 0)
        if struc_component is not None:
            struc_coeffs = np.random.rand(struc_component.shape[1])
            if X is not None: 
                inputs = np.concatenate((inputs, struc_component), axis=1)
                coeffs = np.concatenate((coeffs, struc_coeffs), axis=0)
            else:
                inputs = struc_component
                coeffs = struc_coeffs  
    else:
        d_na = len(idxs_nas)
        if X is not None:
            n, d = X.shape
            d_obs = len(idxs_obs) 
            inputs = X[:, idxs_obs]
            coeffs = np.random.rand(d_obs, d_na)
            coeffs = normalize(inputs, coeffs)
        if struc_component is not None:
            struc_coeffs = np.random.rand(struc_component.shape[1], d_na)
            if X is not None:
                inputs = np.concatenate((inputs, struc_component), axis=1)
                coeffs = np.concatenate((coeffs, struc_coeffs), axis=0)
            else:
                inputs = struc_component
                coeffs = struc_coeffs
    return coeffs, inputs

def fit_intercepts(
        X: np.ndarray,
        coeffs: np.ndarray,
        p_miss: float,
        weak: bool = True,
        self_mask: bool = False
        ) -> np.ndarray:
    
    if self_mask: # MNAR specific implementation
        d = len(coeffs)
        intercepts = np.zeros(d)
        if weak: # probabilistic
            for j in range(d):
                def f(x: np.ndarray) -> np.ndarray:
                    return expit(X*coeffs[j]+x).mean().item() - p_miss # should = 0
                intercepts[j] = optimize.bisect(f, -150, 150) 
        else: # deterministic
            for j in range(d):
                def f(x: np.ndarray) -> np.ndarray:
                    return ((X*coeffs[j]+x) > 0).mean().item() - p_miss # should = 0
                intercepts[j] = optimize.bisect(f, -150, 150) 
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        if weak: # probabilistic 
            for j in range(d_na):
                def f(x: np.ndarray) -> np.ndarray:
                    return expit(np.dot(X, coeffs[:, j])+x).mean().item() - p_miss
                intercepts[j] = optimize.bisect(f, -150, 150) 
        else: # deterministic
            for j in range(d_na):
                def f(x: np.ndarray) -> np.ndarray:
                    return ((np.dot(X, coeffs[:, j])+x) >= 0).mean().item() - p_miss
                intercepts[j] = optimize.bisect(f, -150, 150) 
    return intercepts
 
def MCAR_mask(X: np.ndarray,
              p_miss: float,
              structured: bool=False,
              weak: bool=True,
              sequential: bool=False,
              num_blocks: int= 2) -> np.ndarray:
    if structured == False:
        #print("MCAR Unstructured") 
        mask = np.random.rand(*X.shape) < p_miss # (I)
    else:       
        n, d = X.shape
        mask = np.zeros((n, d)).astype(bool)
        
        if weak and not sequential:
            #print("MCAR Weak + Block")
            blocks = block_assignment(d, num_blocks)
            curr_block = -1
            latent_effects = None
            for j in np.arange(d):
                block = blocks[j]
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X=None, idxs_obs = None, idxs_nas=[j], struc_component=latent_effects, self_mask=False)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()
        
        elif weak and sequential: 
            #print("MCAR Weak + Sequential")
            for j in np.arange(d):
                coeffs, inputs = pick_coeffs(X=None, idxs_obs=None, idxs_nas=[j], struc_component=mask[:, :j], self_mask=False)  # how does this handle the first index? (it does MCAR!)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()
        
        elif not weak and not sequential:
            #print("MCAR Strong + Block")
            blocks = block_assignment(d, num_blocks)
            curr_block = -1
            latent_effects = None
            for j in np.arange(d):
                block = blocks[j]
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], struc_component=(latent_effects), self_mask=False)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = round_near_zero(inputs @ coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
        
        else: 
            #print("MCAR Strong + Sequential")
            for j in np.arange(d): # TODO: spread out the missingness!
                if j == 0:
                    mask[:, j] = np.random.rand(mask.shape[0]) < p_miss
                else:              
                    coeffs, inputs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], struc_component=(mask[:, :j]), self_mask=False) 
                    intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                    ps = round_near_zero(inputs @ coeffs + intercepts) >= 0
                    # print("ps ", ps)
                    mask[:, j] = ps.flatten()
        
    return mask.astype(float)
    
def MAR_mask(X: np.ndarray,
              p_miss: float,
              structured: bool=False,
              weak: bool=True,
              sequential: bool=False,
              p_obs: float=0.5,
              num_blocks: int= 2) -> np.ndarray:
    
    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1)  # number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  # number of variables that will have missing values

    idxs_obs = np.random.choice(d, d_obs, replace=False) # randomly generate indexes of variables with have no missing values
    idxs_nas = np.setdiff1d(np.arange(d), idxs_obs) # indexes of variables that will have missing values
    
    if structured == False:
        if weak: # MAR Probabilistic (VI)
            #print("MAR Probabilistic")
            coeffs, inputs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
            ps = expit(X[:, idxs_obs] @ coeffs + intercepts)
            ber = np.random.rand(n, d_na)
            mask[:, idxs_nas] = ber < ps
        else:
            #print("MAR Deterministic")
            coeffs, inputs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
            ps = (X[:, idxs_obs] @ coeffs + intercepts) >= 0
            mask[:, idxs_nas] = ps
    else:

        if weak and not sequential:
            #print("MAR Weak + Block")
            blocks = block_assignment(d, num_blocks)
            curr_block = -1
            latent_effects = None
            for j in idxs_nas:
                block = blocks[j]
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=latent_effects)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif weak and sequential:
            #print("MAR Weak + Sequential")
            for j in idxs_nas:
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=mask[:, :j])
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif not weak and not sequential:
            #print("MAR Strong + Block")
            blocks = block_assignment(d, num_blocks)
            curr_block = -1
            latent_effects = None
            for j in idxs_nas:
                block = blocks[j]
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=(latent_effects))
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = round_near_zero(inputs @ coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
            
        else:
            #print("MAR Strong + Sequential")
            for j in idxs_nas:
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=(mask[:, :j]))
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = round_near_zero(inputs @ coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
                
    return mask.astype(float)

def MNAR_mask_logistic(
    X: np.ndarray,
    p_miss: float,
    structured: bool = False,
    weak: bool = True,
    sequential: bool = False,
    p_obs: float = 0.3,
    num_blocks: int = 2,
    exclude_inputs: bool = True
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

    idxs_obs = (np.random.choice(d, d_obs, replace=False)) if exclude_inputs else np.arange(d) # randomly generate indexes of variables with have no missing values
    idxs_nas = (np.setdiff1d(np.arange(d), idxs_obs)) if exclude_inputs else np.arange(d) # indexes of variables that will have missing values

    if structured == False:
        if weak:
            #print("MNAR Probabilistic")
            coeffs, inputs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
            ps = expit(X[:, idxs_obs] @ coeffs + intercepts)
            ber = np.random.rand(n, d_na)
            mask[:, idxs_nas] = ber < ps
        else: 
            #print("MNAR Deterministic")
            coeffs, inputs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
            ps = (X[:, idxs_obs] @ coeffs + intercepts) >= 0
            mask[:, idxs_nas] = ps
        
        if exclude_inputs:
            mask[:, idxs_obs] = np.random.rand(n, d_obs) < p_miss
    else:

        if weak and not sequential:
            #print("MNAR Weak + Block")
            blocks = block_assignment(d, num_blocks)
            curr_block = -1
            latent_effects = None
            for j in np.arange(d): 
                block = blocks[j]
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=latent_effects)
                else: # M_obs terms in exclude_inputs = True case
                    coeffs, inputs = pick_coeffs(None, idxs_obs=[], idxs_nas=[j], struc_component=latent_effects, self_mask=False) 
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif weak and sequential: 
            #print("MNAR Weak + Sequential")
            for j in np.arange(d): 
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=mask[:, :j])
                else:
                    coeffs, inputs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], self_mask=False, struc_component=mask[:, :j])
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif not weak and not sequential: 
            #print("MNAR Strong + Block")
            blocks = block_assignment(d, num_blocks)
            curr_block = -1
            latent_effects = None
            for j in np.arange(d): 
                block = blocks[j]
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                if j in idxs_nas: # this is always true for exclude_inputs = False
                    coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=(latent_effects))
                else: #M_obs terms in exclude_inputs = True case
                    coeffs, inputs = pick_coeffs(None, idxs_obs=[], idxs_nas=[j], struc_component=(latent_effects), self_mask=False)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = round_near_zero(inputs @ coeffs + intercepts) >= 0
                mask[:, j] = ps.flatten()
            
        else:
            #print("MNAR Strong + Sequential")
            for j in np.arange(d): 
                if j == 0 and j not in idxs_nas:
                    mask[:, j] = np.random.rand(mask.shape[0]) < p_miss
                else:
                    if j in idxs_nas: # this is always true for exclude_inputs=False
                        coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=(mask[:, :j]))
                    else: # M_obs terms
                        coeffs, inputs = pick_coeffs(X=None, idxs_obs=[], idxs_nas=[j], self_mask=False, struc_component=(mask[:, :j]))
                    intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                    ps = round_near_zero(inputs @ coeffs + intercepts) >= 0
                    mask[:, j] = ps.flatten()

    return mask


def simulate_nan(X: np.ndarray,
                 p_miss: float,
                 mecha: str = "MCAR",
                 opt: str = "logistic",
                 structured: bool=False,
                 weak: bool=True,
                 sequential: bool=False,
                 p_obs: float=0.5,
                 num_blocks: int=2,
                 exclude_inputs: bool=True
                 ) -> np.ndarray:
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, structured, weak, sequential, p_obs, num_blocks)
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, structured, weak, sequential, p_obs, num_blocks, exclude_inputs)
    else:
        mask = MCAR_mask(X, p_miss, structured, weak, sequential, num_blocks)
    X_nas = X.copy()
    X_nas[mask.astype(bool)] = np.nan
    #return X_nas
    return {'X_init': X.astype(np.float64), 'X_incomp': X_nas.astype(np.float64), 'mask': mask}

#np.random.seed(0)
