import numpy as np

def normalize(data, coeffs, tol=1e-10):
    std = np.std(data@coeffs, axis=0, keepdims=True)
    std[np.isclose(std, 0.0, atol=tol)] = 1.0 # TODO: Quantify effects of doing 
    return coeffs / std

def generate_random_cov(n, rng):
    A = rng.standard_normal((n, n))
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

def generate_coeffs(shape: np.ndarray,
                    rng: np.random.Generator, 
                    dist: str="uniform",
                    arg0: float=0.0,
                    arg1: float=1.0):
    if dist == "uniform":
        min = arg0
        max = arg1
        if min >= max:
            raise ValueError("Minimum and maximum endpoints mismatched.")
        if type(shape) == tuple:
            return rng.random(shape) * (max-min) + min
        else:
            return rng.random(shape) * (max- min) + min
    elif dist == "normal":
        mean = arg0
        std = arg1
        return rng.normal(loc=mean, scale=std, size=shape)
    elif dist == "betas":
        beta = next(arg0) if hasattr(arg0, '__next__') else arg0
        return np.full(shape, beta)
    else:
        raise TypeError("Unrecognized coefficient distribution.")
        