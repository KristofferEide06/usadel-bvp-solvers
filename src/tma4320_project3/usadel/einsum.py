import numpy as np
import numpy.typing as npt


def bmm(
    A: npt.NDArray[np.complex128], 
    B: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Matrix multiplication performed in batches using np.einsum
    Args:
        A (npt.NDArray[np.complex128]): Matrix 1, either (2, 2) or (2, 2, N_x)
        B (npt.NDArray[np.complex128]): Matrix 2, either (2, 2) or (2, 2, N_x)
    Returns:
        npt.NDArray[np.complex128]: Product of the matrices
    """

    if A.ndim == 2 and B.ndim == 2:
        return A @ B
    if A.ndim == 2 and B.ndim == 3:
        return np.einsum('ij,jkl->ikl', A, B, optimize=True)
    if A.ndim == 3 and B.ndim == 3:
        return np.einsum('ijk,jlk->ilk', A, B, optimize=True)
    
    raise ValueError(f"Received unknown shapes: {A.shape}, {B.shape}")


def tr(
    A: npt.NDArray[np.complex128]
) -> np.complex128 | npt.NDArray[np.complex128]:
    """Computes the trace of a matrix with np.einsum
    Args:
        A (npt.NDArray[np.complex128]): Matrix to be traced
    Returns:
        np.complex128 | npt.NDArray[np.complex128]: Trace of matrix A, either (2, 2) or (2, 2, N_x)
    """

    if A.ndim == 2:
        return np.einsum('...ii', A, optimize=True)
    
    return np.einsum('iik->k', A, optimize=True)