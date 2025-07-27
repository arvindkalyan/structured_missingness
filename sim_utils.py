import numpy as np

def block_assignment(length, num_blocks):
    q, r = divmod(length, num_blocks)
    # Create block sizes: first r blocks get an extra item
    block_sizes = np.array([q + 1] * r + [q] * (num_blocks - r))
    block_assignments = np.repeat(np.arange(num_blocks), block_sizes)

    block_start_indices = np.insert(np.cumsum(block_sizes[:-1]), 0, 0)
    block_indices = np.arange(length) - block_start_indices[block_assignments]

    return block_assignments, block_indices

def normalize(data, coeffs, tol=1e-10):
    std = np.std(data@coeffs, axis=0, keepdims=True)
    # std[np.isclose(std, 0.0, atol=tol)] = 1.0 # TODO: Quantify effects of doing 
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
        