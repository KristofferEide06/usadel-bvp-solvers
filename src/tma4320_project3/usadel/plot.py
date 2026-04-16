import numpy as np
import numpy.typing as npt
from typing import cast, Literal
from collections.abc import Callable
from scipy.integrate import simpson
import matplotlib.pyplot as plt

from .solver import(
    usadel_solver
)

from .observables import(
    dos_fun,
    current_integrand_fun
)

#Plot 
#real and im parts of sol for gamma for different epsilon
#density of states as function of kposition for different states
#density of states as function of length
#current integral in the middle of metal as function of energy
#current as function of phase difference

def plot_usadel_sol(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    epsilon: float,
    delta: float,
    zeta: float,
    l: float,
    phi_L: float,
    phi_R: float,
    superconductor: bool,
    matrix_label: Literal['gamma', 'gamma_tilde', 'w', 'w_tilde'],
    figsize: tuple[int, ...] = (8,4)
):
    """Plots components of riccati matrix solution

    Args:
        x (npt.NDArray[np.float64]): Array to find solution on
        y (npt.NDARray[np.float64]): Initial guess
        epsilon float: Quasiparticle excitation energy
        delta float: Imaginary energy shift
        zeta float: Interface parameter
        l float: Length of normal region
        phi_L (float): Left superconducting phase.
        phi_R (float): Right superconducting phase.
        superconductor (bool): True if normal metal is interfaced with two superconductors
        matrix_label (str): Which matrix to plot, can be 'gamma', 'gamma_tilde', 'w', 'w_tilde'
        figsize (tuple[int, ...], optional): desired figure size. Defaults to (8,4).

    Raises:
        ValueError: If matrix_label is not amongst available matrices
    """
    
    if matrix_label not in ['gamma', 'gamma_tilde', 'w', 'w_tilde']:
        raise ValueError('argument matrix_label must be gamma, gamma_tilde, w or w_tilde')
        
    gamma, gamma_tilde, w, w_tilde, _ = usadel_solver(
        x, y,
        epsilon, delta, zeta, l,
        phi_L, phi_R,
        superconductor
    )
    
    matrices = {
        "gamma": gamma,
        "gamma_tilde": gamma_tilde,
        "w": w,
        "w_tilde": w_tilde
    }
    
    matrix = matrices[matrix_label]
    matrix_shape = matrix[0].shape
    fig, ax = plt.subplots(*matrix.shape[1:], figsize=figsize, squeeze=False)
    
    for row in range(matrix.shape[1]):
        for col in range(matrix.shape[2]):
            component = matrix[:, row, col]
            
            ax[row,col].plot(x, np.real(component), "r-", label = "Re")
            ax[row][col].plot(x, np.imag(component), "b--", label = "Im")
            ax[row][col].set_xlabel('x')
            ax[row][col].set_ylabel(rf"$\gamma_{{{row}{col}}}$")
            ax[row][col].grid()
            ax[row][col].legend()
        
    plt.tight_layout()
    plt.show()
    
def plot_observable(
    observable: Literal['dos', 'current', 'current_integrand'],
    variable_plot: Literal['x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L', 'phi_R'],
    x: npt.NDArray[np.float64],
    x_index : int,
    y: npt.NDArray[np.float64],
    epsilon: float | npt.NDArray[np.float64],
    delta: float | npt.NDArray[np.float64],
    zeta: float | npt.NDArray[np.float64],
    l: float | npt.NDArray[np.float64],
    phi_L: float | npt.NDArray[np.float64],
    phi_R: float | npt.NDArray[np.float64],
    superconductor: bool,
    figsize: tuple[int, ...] = (8,4)
):
    """Plots observable as function of variable_plot

    Args:
        observable (Literal[&#39;dos&#39;, &#39;current&#39;, &#39;current_integrand&#39;]): observable to plot
        variable_plot (Literal[&#39;x&#39;, &#39;epsilon&#39;, &#39;delta&#39;, &#39;zeta&#39;, &#39;l&#39;, &#39;phi_L&#39;, &#39;phi_R&#39;]): value to plot observable over
        x (npt.NDArray[np.float64]): Array to find solution
        x_index (int): Index of which to evaluate observable if variable_Plot != 'x'
        location of which to evaluate solution, can be any integer if variable_plot == 'x'
        y (npt.NDARray[np.float64]): Initial guess
        epsilon float: Quasiparticle excitation energy
        delta float: Imaginary energy shift
        zeta float: Interface parameter
        l float: Length of normal region
        phi_L (float): Left superconducting phase.
        phi_R (float): Right superconducting phase.
        superconductor (bool): True if normal metal is interfaced with two superconductors
        matrix_label (str): Which matrix to plot, can be 'gamma', 'gamma_tilde', 'w', 'w_tilde'
        figsize (tuple[int, ...], optional): desired figure size. Defaults to (8,4).

    Raises:
        ValueError: if observable not amongst 'docs', 'current', 'current_integrand'
        ValueError: if variable_plot not amongst 'x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L', 'phi_R'
        ValueError: if (variable_plot == 'x' or variable_plot == 'epsilon') and observable == 'current'
    """
    if observable not in ['dos', 'current', 'current_integrand']:
        raise ValueError("observable must be 'dos', 'current', or 'current_integrand'")
    
    if variable_plot not in ['x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L', 'phi_R']:
        raise ValueError("variable_plot must be 'x', 'epsilon', 'delta', 'zeta', 'l', 'phi_L' or 'phi_R'")
    if (variable_plot == 'x' or variable_plot == 'epsilon') and observable == 'current':
        raise ValueError("variable_plot == 'x' or 'epsilon' and observable == 'current' is meaningless combination")
    
  
    variables = {
        'x': x,
        'y': y,
        'epsilon': epsilon,
        'delta': delta,
        'zeta': zeta,
        'l': l,
        'phi_L': phi_L,
        'phi_R': phi_R,
        'superconductor': superconductor
    }
    
    dynamic_var = variables[variable_plot]
    fixed_vars = {k: v for k, v in variables.items() if k!= variable_plot}

    if variable_plot == 'x':
        gamma, gamma_tilde, w, w_tilde, _ = usadel_solver(**variables)

        y_vals = np.zeros(len(x), dtype=np.float64)
        
        if observable == 'dos':
            y_vals = dos_fun(gamma, gamma_tilde)
        elif observable == 'current_integrand':
            y_vals = current_integrand_fun(gamma, gamma_tilde, w, w_tilde)

        x_vals = x
    else:
        dynamic_vals = np.asarray(dynamic_var, dtype = np.float64)
            
        x_vals = dynamic_vals
        y_vals = np.zeros(len(dynamic_vals), dtype=np.float64)
        
        for i, value in enumerate(dynamic_vals):
            current_vars = fixed_vars | {variable_plot: value}

            if observable == 'dos':
                gamma, gamma_tilde, w, w_tilde, _ = usadel_solver(**current_vars)
                y_vals[i] = dos_fun(
                    gamma[x_index],
                    gamma_tilde[x_index],
                )
                
            elif observable == 'current_integrand':
                gamma, gamma_tilde, w, w_tilde, _ = usadel_solver(**current_vars)
                y_vals[i] = current_integrand_fun(
                    gamma[x_index],
                    gamma_tilde[x_index],
                    w[x_index],
                    w_tilde[x_index]
                )    
            elif observable == 'current':
                epsilon_vals = np.linspace(2.0, 0.0, 101)
                integrand_vals = np.zeros_like(epsilon_vals)
                y_current = y.copy()
                
                for j, epsilon_val in enumerate(epsilon_vals):
                    epsilon_vars = current_vars | {'epsilon': epsilon_val}
                    
                    gamma, gamma_tilde, w, w_tilde, sol = usadel_solver(**epsilon_vars)

                    y_current = sol
                    
                    integrand_vals[j] = current_integrand_fun(
                        gamma[x_index],
                        gamma_tilde[x_index],
                        w[x_index],
                        w_tilde[x_index]
                    )

                    epsilon_vars['y'] = y_current
                    
                y_vals[i] = -simpson(np.flip(integrand_vals), x = np.flip(epsilon_vals))
                    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_vals, y_vals)
    if variable_plot == 'x':
        ax.set_title(f"{observable} over {variable_plot}")
    else:
        ax.set_title(f"{observable} over {variable_plot} at x_index = {x_index}")
    ax.set_xlabel(variable_plot)
    ax.set_ylabel(observable)
    ax.grid()
    plt.show()
