import numpy as np
import numpy.typing as npt

def complex_to_real(matrix: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """Transforms real matrix to flattened real vector

    Args:
        matrix (npt.NDArray[np.complex128]): complex matrix

    Returns:
        npt.NDArray[np.float64]: flattened real vector of the form v = (real, imag)
    """
    single = matrix.ndim == 2
    if single:
        matrix = matrix[..., np.newaxis]

    N_x = matrix.shape[-1]
    real_arr = np.real(matrix).reshape(-1, N_x)
    im_arr = np.imag(matrix).reshape(-1, N_x)

    result = np.concatenate([real_arr, im_arr], axis=0, dtype=np.float64)

    if single: 
        return result.squeeze(axis=-1)
    else:
        return result

def real_to_complex(
    vec: npt.NDArray[np.float64], 
    matrix_shape: tuple[int, ...] = (2, 2),
    ) -> npt.NDArray[np.complex128]:
    """Transforms real flattened vector into complex matrix

    Args:
        vec (npt.NDArray[np.float64]): real flattened vector of the form v = (real, imag)
        matrix_shape (tuple[int, ...], optional): Shape of matrix to be returned. Defaults to (2, 2).

    Raises:
        ValueError: vec must be of even size, to have as many complex parts as real

    Returns:
        npt.NDArray[np.complex128]: complex matrix
    """
    single = vec.ndim == 1

    if single:
        vec = vec[:, np.newaxis]

    if vec.shape[0] % 2 != 0:
        raise ValueError("Vec must be of even size")

    n = len(vec)//2
    real_arr = vec[:n]
    im_arr = vec[n:]

    N_x = vec.shape[1]
    result = (real_arr + 1j * im_arr).reshape(*matrix_shape, N_x)

    if single: 
        return result.squeeze(axis=-1)
    else:
        return result

def expand_vec(vec_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Expands array of real vectors (matrix) to one vector

    Args:
        vec_arr (npt.NDArray[np.float64]): array of vectors to expand

    Returns:
        npt.NDArray[np.float64]: expanded vector
    """
    
    return np.concat(vec_arr, axis = 0, dtype = np.float64)

def reshape_vec(
    vec: npt.NDArray[np.float64], 
    vec_size: int = 8,
    ) -> npt.NDArray[np.float64]:
    """Creates component vectors from one vector
    
    Args:
        vec (npt.NDArray[np.float64]): vector to be reshaped.
        vec_size (int, optional): desired size of component vectors. Defaults to 8

    Raises:
        ValueError: Vector must be divisible by desired vector component length

    Returns:
        npt.NDArray[np.float64]: matrix where each row is component vector of original vector
    """
    if len(vec) % vec_size != 0:
        raise ValueError("Vec must be divisible by vec_size")

    return vec.reshape(-1, vec_size)

def usadel_matrix_to_vec(
    gamma: npt.NDArray[np.complex128], 
    gamma_tilde: npt.NDArray[np.complex128],
    w: npt.NDArray[np.complex128],
    w_tilde: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.float64]:
    """Expands the four usadel matrices into one real vector

    Args:
        gamma (npt.NDArray[np.complex128]): gamma matrix
        gamma_tilde (npt.NDArray[np.complex128]): gamma tilde matrix
        w (npt.NDArray[np.complex128]): w matrix
        w_tilde (npt.NDArray[np.complex128]): w tilde matrix

    Raises:
        ValueError: all matrices should be of shape (2,2)

    Returns:
        npt.NDArray[np.float64]: expanded real vector
    """
    single = gamma.ndim == 2

    if single:
        gamma, gamma_tilde, w, w_tilde = (
            m[..., np.newaxis] for m in [gamma, gamma_tilde, w, w_tilde]
        )

    if not (gamma.shape == gamma_tilde.shape == w.shape == w_tilde.shape):
        raise ValueError("All matrices must have the same shape")

    result = np.stack([
        complex_to_real(gamma),
        complex_to_real(gamma_tilde),
        complex_to_real(w),
        complex_to_real(w_tilde)
    ], axis=0).reshape(-1, gamma.shape[-1]) 

    if single:
        return result.squeeze(axis=-1)
    else:
        return result
    
def vec_to_usadel_matrix(
    vec: npt.NDArray[np.float64],
    matrix_shape: tuple[int, ...] = (2, 2),
    ) -> npt.NDArray[np.complex128]:
    """Transform vector into the usadel matrices, gamma, gamma_tilde, w, w_tilde

    Args:
        vec (npt.NDArray[np.float64]): vector to be transformed
        matrix_shape (tuple[int, ...], optional): shape of usadel matrices. Defaults to (2, 2)

    Raises:
        ValueError: Matrix_shape should be so that vec can be divided into equally shaped matrices

    Returns:
        npt.NDArray[np.complex128]: array of matrices, where the columns represent gamma, gamma_tilde, w, w_tilde respectively
    """
    single = vec.ndim == 1

    if single:
        vec = vec[:, np.newaxis]

    component_size = np.prod(matrix_shape)
    N_x = vec.shape[1]

    if vec.shape[0] % component_size != 0:
        raise ValueError("vector size and matrix_shape not compatible")

    component_vecs = vec.reshape(component_size, -1, N_x)
    result = np.array([
        real_to_complex(component_vecs[i], matrix_shape)
        for i in range(component_size)
    ])

    if single:
        return result.squeeze(axis=-1)
    else:
        return result
