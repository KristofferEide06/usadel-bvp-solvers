import numpy as np
import numpy.typing as npt


def bmm(
    A: npt.NDArray[np.complex128], 
    B: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Matrix multiplication preformed in batches using np.einsum

    Args:
        A (npt.NDArray[np.complex128]): Matrix 1 (2, 2, N_x)
        B (npt.NDArray[np.complex128]): Matrix 2 (2, 2, N_x)

    Returns:
        npt.NDArray[np.complex128]: Product of the matrices
    """

    assert A.shape == B.shape, f"A og B må ha samme form, fikk {A.shape} og {B.shape}"
    assert A.shape[0] == A.shape[1] == 2, f"Forventet (2, 2, N_x), fikk {A.shape}"
    
    return np.einsum('ijk,jlk->ilk', A, B, optimize=True)


def tr(
    A: npt.NDArray[np.complex128]
) -> np.complex128:
    """Computes the trace of a matrix with np.einsum

    Args:
        A (npt.NDArray[np.complex128]): Matrix to be traced

    Returns:
        np.complex128: Trace of matrix A
    """

    return np.einsum('...ii', A, optimize=True)


