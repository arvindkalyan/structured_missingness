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
    ) -> np.ndarray:
    n, d = X.shape
    if self_mask:
        coeffs = np.random.rand(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.rand(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs # multiply so we can get std. dev
        coeffs /= np.std(Wx, 0, keepdims=True) # normalize with std dev
    return coeffs

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
 
def MCAR_mask(X: np.ndarray,
              p_miss: float,
              structured: bool=False,
              weak: bool=True,
              sequential: bool=False) -> np.ndarray:
    if structured == False:
        print("MCAR Unstructured") 
        mask = np.random.rand(*X.shape) < p_miss # (I)
    else:        
        if weak and not sequential: #TODO: MCAR Weak + Block (II)
            print("MCAR Weak + Block")
            pass
        elif weak and sequential: #TODO: MCAR Weak + Sequential (IV)
            print("MCAR Weak + Sequential")
            pass
        elif not weak and not sequential: #TODO: MCAR Strong + Block (III)
            print("MCAR Strong + Block")
            pass
        else: #TODO: MCAR Strong + Sequential (V)
            print("MCAR Strong + Sequential")
            pass
        
    return mask.astype(float)
    
def MAR_mask(X: np.ndarray,
              p_miss: float,
              structured: bool=False,
              weak: bool=True,
              sequential: bool=False,
              p_obs: float=0.5) -> np.ndarray:
    
    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1)  # number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  # number of variables that will have missing values

    idxs_obs = np.random.choice(d, d_obs, replace=False) # randomly generate indexes of variables with have no missing values
    idxs_nas = np.setdiff1d(np.arange(d), idxs_obs) # indexes of variables that will have missing values
    
    if structured == False:
        if weak: # MAR Probabilistic (VI)
            print("MAR Probabilistic")
            coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p_miss, weak)
            ps = expit(X[:, idxs_obs] @ coeffs + intercepts)
            ber = np.random.rand(n, d_na)
            mask[:, idxs_nas] = ber < ps
        else: # TODO: MAR Deterministic (VII)
            print("MAR Deterministic")
            coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
            intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p_miss, weak)
            ps = (X[:, idxs_obs] @ coeffs + intercepts) > 0 #TODO: check signs on ps
            mask[:, idxs_nas] = ps
    else:
        if weak and not sequential: #TODO: MAR Weak + Block (VIII)
            print("MAR Weak + Block")
            pass
        elif weak and sequential: #TODO: MAR Weak + Sequential (X)
            print("MAR Weak + Sequential")
            pass
        elif not weak and not sequential: #TODO: MAR Strong + Block (IX)
            print("MAR Strong + Block")
            pass
        else: #TODO: MAR Strong + Sequential (XI)
            print("MAR Strong + Sequential")
            pass

    return mask.astype(float)

def simulate_nan(X: np.ndarray,
                 p_miss: float,
                 mecha: str = "MCAR",
                 structured: bool=False,
                 weak: bool=True,
                 sequential: bool=False,
                 p_obs: float=0.5) -> np.ndarray:
    if mecha == "MCAR":
        mask = MCAR_mask(X, p_miss, structured, weak, sequential)
    elif mecha == "MAR":
        mask = MAR_mask(X, p_miss, structured, weak, sequential, p_obs)
    X_nas = X.copy()
    X_nas[mask.astype(bool)] = np.nan
    return X_nas

X = np.random.rand(100, 100)
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

X_nas = simulate_nan(X, 0.5, mecha="MAR", weak=True)
print(X_nas)
print(X)
X_nas = simulate_nan(X, 0.5, mecha="MAR", weak=False)
print(X_nas)
print(X)