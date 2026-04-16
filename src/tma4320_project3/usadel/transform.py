import numpy as np
import numpy.typing as npt
from typing import cast
from collections.abc import Callable

def complex_to_real(matrix: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """Transforms real matrix to flattened real vector

    Args:
        matrix (npt.NDArray[np.complex128]): complex matrix

    Returns:
        npt.NDArray[np.float64]: flattened real vector of the form v = (real, imag)
    """
    real_arr = np.real(matrix).flatten()
    im_arr = np.imag(matrix).flatten()
    
    return np.concat((real_arr, im_arr), axis = 0, dtype = np.float64)

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
    
    if len(vec) % 2 != 0:
        raise ValueError("Vec must be of even size")
    
    n = len(vec)//2
    real_arr = vec[:n]
    im_arr = vec[n:]
    
    return (real_arr + 1j * im_arr).reshape(matrix_shape)

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

    return vec.reshape(vec_size, -1)

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
    if not (gamma.shape == gamma_tilde.shape == w.shape == w_tilde.shape == (2, 2)):
        raise ValueError("All matrix shapes should be of (2,2)")

    return expand_vec(np.array([
      complex_to_real(gamma),
      complex_to_real(gamma_tilde),
      complex_to_real(w),
      complex_to_real(w_tilde)  
    ]))
    
def vec_to_usadel_matrix(
    vec: npt.NDArray[np.float64],
    matrix_shape: tuple[int, ...] = (2, 2)
    ) -> npt.NDArray[np.complex128]:
    """Transform vector into the form usadel matrices, gamma, gamma_tilde, w, w_tilde

    Args:
        vec (npt.NDArray[np.float64]): vector to be transformed
        matrix_shape (tuple[int, ...], optional): shape of usadel matrices. Defaults to (2, 2)

    Raises:
        ValueError: Matrix_shape should be so that vec can be divided into equally shaped matrices

    Returns:
        npt.NDArray[np.complex128]: array of matrices, where the columns represent gamma, gamma_tilde, w, w_tilde respectively
    """
    component_size = np.prod(matrix_shape)
    if len(vec) % component_size != 0:
        raise ValueError("vector size and matrix_shape not compatible")
    
    component_vec_arr = vec.reshape(component_size, -1)
    
    return np.array([real_to_complex(component_vec, matrix_shape) for component_vec in component_vec_arr])

