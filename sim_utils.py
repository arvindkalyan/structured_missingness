import numpy as np

def block_assignment(length, num_blocks):
    q, r = divmod(length, num_blocks)
    # Create block sizes: first `r` blocks get an extra item
    block_sizes = np.array([q + 1] * r + [q] * (num_blocks - r))
    return np.repeat(np.arange(num_blocks), block_sizes)

def normalize(data, coeffs, tol=1e-10):
    std = np.std(data@coeffs, axis=0, keepdims=True)
    # std[np.isclose(std, 0.0, atol=tol)] = 1.0 # TODO: Quantify effects of doing 
    return coeffs / std

def random_flip(mask: np.ndarray, flip_intensity: float = 0.5) -> np.ndarray:
    # return mask
    A = mask    
    one_mask = mask==1
    zero_mask = ~one_mask
    num_1 = np.sum(one_mask)
    num_0 = mask.size - num_1
    if num_1 == 0 or num_0 == 0 or flip_intensity == 0:
        return mask

    k_target = flip_intensity * min(num_1, num_0)
    p_10 = k_target / num_1
    p_01 = k_target / num_0
    rand = np.random.rand(*mask.shape)
    flip_to_0 = (one_mask) & (rand < p_10)
    flip_to_1 = (zero_mask) & (rand < p_01)
    B = np.where(flip_to_0, 0, np.where(flip_to_1, 1, mask))

    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ B
    AX = A @ X
    # print("multiplied with: ", X)
    # print("apprx: ", AX)
    # print("actual: ", B)
    return np.where(flip_to_0, 0, np.where(flip_to_1, 1, mask))

def random_negate(mask: np.ndarray, flip_intensity: float = 0.5) -> np.ndarray:
    # return mask
    A = mask

    pos_mask = mask >= 0
    neg_mask = ~pos_mask
    num_pos = np.sum(pos_mask)
    num_neg = mask.size - num_pos
    if num_pos == 0 or num_neg == 0 or flip_intensity == 0:
        return mask

    k_target = flip_intensity * min(num_pos, num_neg)
    p_10 = k_target / num_pos
    p_01 = k_target / num_neg
    rand = np.random.rand(*mask.shape)
    flip_mask = (pos_mask & (rand < p_10)) | (neg_mask & (rand < p_01))
    B = np.where(flip_mask, -mask, mask)

    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ B
    AX = A @ X
    # print("multiplied with: ", X)
    # print("apprx: ", AX)
    # print("actual: ", B)
    return np.where(flip_mask, -mask, mask)

def round_near_zero(arr: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    assert(tolerance>0)
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
        