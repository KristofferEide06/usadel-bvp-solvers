import numpy as np
import numpy.typing as npt
from typing import cast
from collections.abc import Callable
    
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
    if not (gamma.shape == gamma_tilde.shape == (2, 2)):
        raise ValueError("gamma and gamma_tilde should be of size (2,2)")
    
    I = np.eye(gamma.shape[0], dtype = np.complex128)
    
    N = cast(npt.NDArray[np.complex128], np.linalg.inv(I - gamma @ gamma_tilde))
    N_tilde = cast(npt.NDArray[np.complex128], np.linalg.inv(I - gamma_tilde @ gamma))

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
    
    d_N = N @ (w @ gamma_tilde + gamma @ w_tilde) @ N
    d_N_tilde = N_tilde @ (w_tilde @ gamma + gamma_tilde @ w) @ N_tilde
    
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
        phi_L (float, optional): Left superconducting phase. 
        phi_R (float, optional): Right superconducting phase. 
        
    Returns:
        npt.NDArray[np.complex128]: Array of matrices gamma_L, gamma_L_tilde, gamma_R, gamma_R_tilde respectively
    """
    
    vartheta = lambda sigma, epsilon = epsilon, delta = delta: np.atanh(sigma/(epsilon + 1j * delta))
    s = lambda sigma, epsilon = epsilon, delta = delta: np.sinh(vartheta(epsilon, delta, sigma))
    c = lambda sigma, epsilon = epsilon, delta = delta: np.cosh(vartheta(epsilon, delta, sigma))

    gamma_L = np.matrix([0, s(1) / (1 + c(1))], [s(-1)/(1 + c(-1)), 0]) * np.exp(-1j * phi_L)
    gamma_L_tilde = np.matrix([0, s(1) / (1 + c(-1))], [s(1) / (1 + c(1)), 0]) * np.exp(-1j * phi_L)
    gamma_R = np.matrix([0, s(1) / (1 + c(1))], [s(-1) / (1 + c(-1)), 0]) * np.exp(-1j * phi_R)
    gamma_R_tilde = np.matrix([0, s(1) / (1 + c(-1))], [s(1) / (1 + c(1)), 0]) * np.exp(-1j * np.exp(-1j * phi_R))
    
    return np.array([gamma_L, gamma_L_tilde, gamma_R, gamma_R_tilde])

def rho_3_fun(matrix_shape: tuple[int, ...] = (2, 2)) -> npt.NDArray[np.complex128]: #consider removing generalization
    """Calculates the Pauli-z in nambu space

    Args:
        matrix_shape (tuple[int, ...], optional): Shape of I matrix, should always be 2x2 for the pauli matrix. Defaults to (2, 2).

    Returns:
        npt.NDArray[np.complex128]: Pauli-z in nambu space
    """
    I = np.eye(matrix_shape[0])
    
    return np.matrix([I, np.zeros_like(I)], [np.zeros_like(I), -I])

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
    I = np.eye(gamma.shape[0])
    
    return np.matrix([2 * N - I, 2 * N @ gamma], [-2 * N_tilde @ gamma_tilde, -2 * N_tilde + I]) 

      
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
    N, N_tilde = N_fun(gamma, gamma_tilde)
    d_N, d_N_tilde = N_deriv_fun(gamma, gamma_tilde)
    
    return 2 * np.matrix(
        [d_N, N @ w + d_N @ gamma], 
        [-N_tilde @ w_tilde - d_N_tilde @ gamma_tilde, -d_N_tilde]
        )

