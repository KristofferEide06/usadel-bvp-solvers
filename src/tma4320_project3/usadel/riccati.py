import numpy as np
import numpy.typing as npt
from typing import cast
from collections.abc import Callable

from .einsum import(
    bmm
)
    
def N_fun(
    gamma: npt.NDArray[np.complex128], 
    gamma_tilde: npt.NDArray[np.complex128]
    ) -> tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128]
]:
    """Calculates the Riccati normalization matrices

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): conjugate Riccati amplitude

    Raises:
        ValueError: gamma and gamma_tilde should be of size (2,2)

    Returns:
        tuple[ npt.NDArray[np.complex128], npt.NDArray[np.complex128] ]: normalization matrix), conjugate normalization matrix
    """
    single = gamma.ndim == 2

    if single:
        gamma = gamma[..., np.newaxis]
        gamma_tilde = gamma_tilde[..., np.newaxis]

    if gamma.shape[:2] != (2, 2) or gamma.shape != gamma_tilde.shape:
        raise ValueError("gamma and gamma_tilde should have shape (2,2) or (2,2,N_x)")
    
    N_x = gamma.shape[-1]
    I = np.eye(2, dtype=np.complex128)[..., np.newaxis] * np.ones(N_x)

    A = np.moveaxis(I - bmm(gamma, gamma_tilde), -1, 0)
    B = np.moveaxis(I - bmm(gamma_tilde, gamma), -1, 0)
    
    N = cast(npt.NDArray[np.complex128], np.moveaxis(np.linalg.inv(A), 0, -1))
    N_tilde = cast(npt.NDArray[np.complex128], np.moveaxis(np.linalg.inv(B), 0, -1))

    if single:
        return N.squeeze(axis=-1), N_tilde.squeeze(axis=-1)
    else:
        return N, N_tilde

def N_deriv_fun(
    gamma: npt.NDArray[np.complex128],
    gamma_tilde: npt.NDArray[np.complex128],
    w: npt.NDArray[np.complex128],
    w_tilde: npt.NDArray[np.complex128]
    ) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128]
]:
    """Calculates the derivative of the normalization matrix

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): Conjugate Riccati amplitude
        w (npt.NDArray[np.complex128]): d_x gamma
        w_tilde (npt.NDArray[np.complex128]): d_x gamma_tilde

    Returns:
        tuple[ npt.NDArray[np.complex128], npt.NDArray[np.complex128] ]: d_x N, d_x N_tilde
    """
    N, N_tilde = N_fun(gamma, gamma_tilde)

    d_N = bmm(bmm(N, bmm(w, gamma_tilde) + bmm(gamma, w_tilde)), N)
    d_N_tilde = bmm(bmm(N_tilde, bmm(w_tilde, gamma) + bmm(gamma_tilde, w)), N_tilde)

    return d_N, d_N_tilde  

def Riccati_superconductor(
    epsilon: float,
    delta: float,
    phi_L: float,
    phi_R: float
) -> npt.NDArray[np.complex128]:
    """Calculates Riccati boundary matrices for normal metal insterfaced with two superconductors

    Args:
        epsilon (float): Quasiparticle excitation energy
        delta (float): Imaginary energy shift
        phi_L (np.float64, optional): Left superconducting phase. 
        phi_R (np.float64, optional): Right superconducting phase. 
        
    Returns:
        npt.NDArray[np.complex128]: Array of matrices gamma_L, gamma_L_tilde, gamma_R, gamma_R_tilde respectively
    """
    
    vartheta = lambda sigma: np.atanh(sigma / (epsilon + 1j * delta))
    s = lambda sigma: np.sinh(vartheta(sigma))
    c = lambda sigma: np.cosh(vartheta(sigma))

    gamma_L = np.block([[0, s(1) / (1 + c(1))], [s(-1)/(1 + c(-1)), 0]])* np.exp(-1j * phi_L)
    gamma_L_tilde = np.block([[0, s(1) / (1 + c(-1))], [s(1) / (1 + c(1)), 0]]) * np.exp(-1j * phi_L)
    gamma_R = np.block([[0, s(1) / (1 + c(1))], [s(-1)/(1 + c(-1)), 0]])* np.exp(-1j * phi_R)
    gamma_R_tilde = np.block([[0, s(1) / (1 + c(-1))], [s(1) / (1 + c(1)), 0]]) * np.exp(-1j * phi_R)
    
    return np.array([gamma_L, gamma_L_tilde, gamma_R, gamma_R_tilde])

def rho_3_fun(matrix_shape: tuple[int, ...] = (2, 2)) -> npt.NDArray[np.complex128]: #consider removing generalization
    """Calculates the Pauli-z in nambu space

    Args:
        matrix_shape (tuple[int, ...], optional): Shape of I matrix, should always be 2x2 for the pauli matrix. Defaults to (2, 2).

    Returns:
        npt.NDArray[np.complex128]: Pauli-z in nambu space
    """
    I = np.eye(matrix_shape[0], dtype = np.complex128)
    
    return np.block([[I, np.zeros_like(I)], [np.zeros_like(I), -I]])

def green_fun(
    gamma: npt.NDArray[np.complex128],
    gamma_tilde: npt.NDArray[np.complex128]
)->  npt.NDArray[np.complex128]:
    """Calculates Green function

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): Conjugate Riccati amplitude

    Returns:
        npt.NDArray[np.complex128]: Green function
    """
    N, N_tilde = N_fun(gamma, gamma_tilde)
    single = gamma.ndim == 2

    if single:
        I = np.eye(2, dtype=np.complex128)
        return np.block([
            [2*N - I, 2 * N @ gamma],
            [-2 * N_tilde @ gamma_tilde, -2*N_tilde + I]
        ])
    else:
        N_x = gamma.shape[-1]
        I = np.eye(2, dtype=np.complex128)[..., np.newaxis] * np.ones(N_x)

        first_row = np.concatenate([2*N - I, 2*bmm(N, gamma)], axis=1)
        second_row = np.concatenate([-2*bmm(N_tilde, gamma_tilde), -2*N_tilde + I], axis=1)

        return np.concatenate([first_row, second_row], axis=0)

      
def green_fun_deriv(
    gamma: npt.NDArray[np.complex128],
    gamma_tilde: npt.NDArray[np.complex128],
    w: npt.NDArray[np.complex128],
    w_tilde: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Calculates the derivative of the Green function

    Args:
        gamma (npt.NDArray[np.complex128]): Riccati amplitude
        gamma_tilde (npt.NDArray[np.complex128]): conjugate Riccati amplitude
        w (npt.NDArray[np.complex128]): d_x gamma
        w_tilde (npt.NDArray[np.complex128]): d_x gamma_tilde

    Returns:
        npt.NDArray[np.complex128]: d_x g
    """
    N, N_tilde   = N_fun(gamma, gamma_tilde)
    d_N, d_N_tilde = N_deriv_fun(gamma, gamma_tilde, w, w_tilde)
    single = gamma.ndim == 2

    if single:
        return 2 * np.block([
            [d_N, N @ w + d_N @ gamma],
            [-N_tilde @ w_tilde - d_N_tilde @ gamma_tilde, -d_N_tilde]
        ])
    else:
        first_row = np.concatenate([2*d_N, 2*(bmm(N, w) + bmm(d_N, gamma))], axis=1)
        second_row = np.concatenate([
            -2*(bmm(N_tilde, w_tilde) + bmm(d_N_tilde, gamma_tilde)),
            -2*d_N_tilde
        ], axis=1)

        return np.concatenate([first_row, second_row], axis=0)

