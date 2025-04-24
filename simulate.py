# stdlib
from typing import List

# third party
import numpy as np
from scipy import optimize
from scipy.special import expit

def pick_coeffs(
    X: np.ndarray,
    idxs_obs: List[int] = [],
    idxs_nas: List[int] = [],
    self_mask: bool = False,
    struc_component: np.ndarray = None
    ) -> np.ndarray:
    n, d = X.shape
    if self_mask: # MNAR specific implementation
        inputs = X
        if struc_component is not None:
            inputs = np.concatenate((inputs, struc_component), axis=1)
            d += struc_component.shape[1]
        coeffs = np.random.rand(d)
        Wx = inputs * coeffs
        coeffs /= np.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        inputs = X[:, idxs_obs]
        if struc_component is not None:
            inputs = np.concatenate((inputs, struc_component), axis=1)
            d_obs += struc_component.shape[1]
        coeffs = np.random.rand(d_obs, d_na)
        
        Wx = inputs @ coeffs # multiply so we can get std. dev
        std = np.std(Wx, 0, keepdims=True)
        std[std == 0] = 1.0 # avoid division by zero
        coeffs /= std # normalize with std dev
    return coeffs, inputs

def fit_intercepts(
        X: np.ndarray,
        coeffs: np.ndarray,
        p_miss: float,
        weak: bool = True,
        self_mask: bool = False
        ) -> np.ndarray:
    
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        if weak: # probabilistic
            for j in range(d):
                def f(x: np.ndarray) -> np.ndarray:
                    return expit(X*coeffs[j]+x).mean().item() - p_miss # should = 0
                intercepts[j] = optimize.bisect(f, -50, 50) #TODO: Better way to implement this?
        else: # deterministic
            for j in range(d):
                def f(x: np.ndarray) -> np.ndarray:
                    return ((X*coeffs[j]+x) > 0).mean().item() - p_miss # should = 0
                intercepts[j] = optimize.bisect(f, -50, 50) #TODO: Better way to implement this?
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        if weak: # probabilistic 
            for j in range(d_na):
                def f(x: np.ndarray) -> np.ndarray:
                    return expit(np.dot(X, coeffs[:, j])+x).mean().item() - p_miss
                intercepts[j] = optimize.bisect(f, -50, 50) #TODO: Better way to implement this?
        else: # deterministic
            for j in range(d_na):
                def f(x: np.ndarray) -> np.ndarray:
                    return ((np.dot(X, coeffs[:, j])+x) > 0).mean().item() - p_miss
                intercepts[j] = optimize.bisect(f, -50, 50) #TODO: Better way to implement this?
    return intercepts
 
def create_block_effects(
        block_size: int,
        n: int) -> np.ndarray:
    
    num_effects = int((block_size-1)*(block_size)/2)
    A = np.random.rand(num_effects, num_effects) # covariance matrix
    cov = np.dot(A, A.T) # covariance matrix
    mean = np.zeros(num_effects) # mean vector
    block_effect = np.random.multivariate_normal(mean=mean, cov=cov, size=n) # latent variable containing effects from neighboring [block_size] elements

    # X_obs mask
    # coeffs, inputs = pick_coeffs(X_obs, np.arange(X_obs.shape[1]), np.arange(block_size))
    # intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
    # block_effect = np.dot(inputs, coeffs) + intercepts

    mat = np.zeros((block_size, block_size-1, n))
    mat[np.triu_indices(block_size-1)] = block_effect.T
    mat[np.tril_indices(block_size, k=-1)] = np.transpose(mat, (1, 0, 2))[np.tril_indices(block_size-1)]
    return mat

def MCAR_mask(X: np.ndarray,
              p_miss: float,
              structured: bool=False,
              weak: bool=True,
              sequential: bool=False,
              num_blocks: int= 2) -> np.ndarray:
    if structured == False:
        print("MCAR Unstructured") 
        mask = np.random.rand(*X.shape) < p_miss # (I)
    else:       
        n, d = X.shape
        mask = np.zeros((n, d)).astype(bool)
        
        if weak and not sequential: #TODO: MCAR Weak + Block (II)
            print("MCAR Weak + Block")
            block_size = (d // num_blocks) + 1
            curr_block = -1
            latent_effects = None
            for j in np.arange(d):
                block = j // block_size
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X, [], [j], self_mask=False, struc_component=latent_effects)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()
        
        elif weak and sequential: #TODO: MCAR Weak + Sequential (IV)
            print("MCAR Weak + Sequential")
            for j in np.arange(d):
                coeffs, inputs = pick_coeffs(X, [], [j], self_mask=False, struc_component=mask[:, :j]) 
                #coeffs, inputs = pick_coeffs(mask, np.arange(j), [j], self_mask=False)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()
        
        elif not weak and not sequential: #TODO: MCAR Strong + Block (III)
            print("MCAR Strong + Block")
            block_size = (d // num_blocks) + 1
            curr_block = -1
            latent_effects = None
            for j in np.arange(d):
                block = j // block_size
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X, [], [j], self_mask=False, struc_component=latent_effects)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = (inputs @ coeffs + intercepts) > 0
                mask[:, j] = ps.flatten()
        
        else: #TODO: MCAR Strong + Sequential (V)
            print("MCAR Strong + Sequential")
            for j in np.arange(d):
                coeffs, inputs = pick_coeffs(X, [], [j], self_mask=False, struc_component=mask[:, :j]) 
                #coeffs, inputs = pick_coeffs(mask, np.arange(j), [j], self_mask=False)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = (inputs @ coeffs + intercepts) > 0
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
            print("MAR Probabilistic")
            coeffs, inputs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
            ps = expit(X[:, idxs_obs] @ coeffs + intercepts)
            ber = np.random.rand(n, d_na)
            mask[:, idxs_nas] = ber < ps
        else: # TODO: MAR Deterministic (VII)
            print("MAR Deterministic")
            coeffs, inputs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
            ps = (X[:, idxs_obs] @ coeffs + intercepts) > 0 #TODO: check signs on ps
            mask[:, idxs_nas] = ps
    else:

        if weak and not sequential: #TODO: MAR Weak + Block (VIII)
            print("MAR Weak + Block")
            block_size = (d // num_blocks) + 1
            curr_block = -1
            latent_effects = None
            for j in idxs_nas:
                block = j // block_size
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=latent_effects)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif weak and sequential: #TODO: MAR Weak + Sequential (X)
            print("MAR Weak + Sequential")
            for j in idxs_nas:
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=mask[:, :j])
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = expit(inputs @ coeffs + intercepts)
                ber = np.random.rand(n, 1)
                mask[:, j] = (ber < ps).flatten()

        elif not weak and not sequential: #TODO: MAR Strong + Block (IX)
            print("MAR Strong + Block")
            block_size = (d // num_blocks) + 1
            curr_block = -1
            latent_effects = None
            for j in idxs_nas:
                block = j // block_size
                if block != curr_block:
                    curr_block = block
                    latent_effects = np.random.randn(n, 1) # latent variable containing effects from [block_size] elements
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=latent_effects)
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = (inputs @ coeffs + intercepts) > 0
                mask[:, j] = ps.flatten()
            
            
        else: #TODO: MAR Strong + Sequential (XI)
            print("MAR Strong + Sequential")
            for j in idxs_nas:
                coeffs, inputs = pick_coeffs(X, idxs_obs, [j], self_mask=False, struc_component=mask[:, :j])
                intercepts = fit_intercepts(inputs, coeffs, p_miss, weak)
                ps = (inputs @ coeffs + intercepts) > 0
                mask[:, j] = ps.flatten()
                

    return mask.astype(float)

def simulate_nan(X: np.ndarray,
                 p_miss: float,
                 mecha: str = "MCAR",
                 structured: bool=False,
                 weak: bool=True,
                 sequential: bool=False,
                 p_obs: float=0.5,
                 num_blocks: int=1) -> np.ndarray:
    if mecha == "MCAR":
        mask = MCAR_mask(X, p_miss, structured, weak, sequential, num_blocks)
    elif mecha == "MAR":
        mask = MAR_mask(X, p_miss, structured, weak, sequential, p_obs, num_blocks)
    X_nas = X.copy()
    X_nas[mask.astype(bool)] = np.nan
    #return X_nas
    return {'X_init': X.astype(np.float64), 'X_incomp': X_nas.astype(np.float64), 'mask': mask}

#np.random.seed(0)

#X = np.random.rand(100, 100)
#X = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=float)

# X_nas = simulate_nan(X, 0.5, mecha="MAR", structured=True, weak=True, sequential=False)
# print("INCL ", X_nas['X_incomp'])
# print("INIT ", X_nas['X_init'])
# print("MASK ", X_nas['mask'])

# X_nas = simulate_nan(X, 0.5, mecha="MAR", structured=True, weak=False, sequential=False)
# print("INCL ", X_nas['X_incomp'])
# print("INIT ", X_nas['X_init'])
# print("MASK ", X_nas['mask'])

# X_nas = simulate_nan(X, 0.5, mecha="MCAR", structured=True, weak=True, sequential=False)
# print("INCL ", X_nas['X_incomp'])
# print("INIT ", X_nas['X_init'])
# print("MASK ", X_nas['mask'])

# X_nas = simulate_nan(X, 0.5, mecha="MCAR", structured=True, weak=False, sequential=False)
# print("INCL ", X_nas['X_incomp'])
# print("INIT ", X_nas['X_init'])
# print("MASK ", X_nas['mask'])

'''
X_nas = simulate_nan(X, 0.5, mecha="MAR", weak=True)
print(X_nas)
print(X)

X_nas = simulate_nan(X, 0.5, mecha="MAR", weak=False)
print(X_nas)
print(X)

X_nas = simulate_nan(X, 0.5, mecha="MAR", structured=True, weak=True, sequential=False)
print(X_nas)
print(X)

X_nas = simulate_nan(X, 0.5, mecha="MAR", structured=True, weak=True, sequential=True)
print(X_nas)
print(X)

X_nas = simulate_nan(X, 0.5, mecha="MAR", structured=True, weak=False, sequential=False)
print(X_nas)
print(X)

X_nas = simulate_nan(X, 0.5, mecha="MAR", structured=True, weak=False, sequential=True)
print(X_nas)
print(X)
'''