import numpy as np
from numba.typed import List
from numba import types
import numpy.typing as npt

from .ivp_solver import (
    IVP_solver_1,
    IVP_solver_2,
)

dict_type = types.DictType(types.unicode_type, types.float64)

def g(s: float) -> float:
    """defines the scalar function used in the secant test.
    
    Args:
    s: the input value

    Returns:
    g_s: the function value at s
    """
    return s + np.sin(s) + np.cos(s)

def sekant_root_g(
        z_0:float,
        z_1:float, 
        params: dict[str, float]) -> tuple[float, int]:
    """finds a root of g with the secant method.
    
    Args:
    z_0: the first initial guess
    z_1: the second initial guess
    params: a dictionary of given parameters

    Returns:
    z[n]: the approximated root
    n: the number of iterations
    """
    z = List.empty_list(types.float64)
    z.append(z_0)
    z.append(z_1)
    n=1
    while abs(z[n]-z[n-1])>params["tol"]:
        z.append((z[n-1]*g(z[n])-z[n]*g(z[n-1]))/(g(z[n])-g(z[n-1])))
        n+=1
    return z[n],n

def sekant_root_1(z_0:float, z_1:float, params: dict[str, float]) -> tuple[float, int]:
    """finds the shooting slope for the first system with the secant method.
    
    Args:
    z_0: the first initial guess
    z_1: the second initial guess
    params: a dictionary of given parameters

    Returns:
    z[n]: the approximated shooting slope
    n: the number of iterations
    """
    z = List.empty_list(types.float64)
    z.append(z_0)
    z.append(z_1)
    n=1
    while abs(z[n-1]-z[n])>params["tol"]:
        z.append((z[n-1]*F_1(z[n],params)-z[n]*F_1(z[n-1],params))/(F_1(z[n],params)-F_1(z[n-1],params)))
        n+=1
    return z[n],n

def sekant_root_2(z_0:float, z_1:float, params: dict[str, float]) -> tuple[float, int]:
    """finds the shooting slope for the second system with the secant method.
    
    Args:
    z_0: the first initial guess
    z_1: the second initial guess
    params: a dictionary of given parameters

    Returns:
    z[n]: the approximated shooting slope
    n: the number of iterations
    """
    z = List.empty_list(types.float64)
    z.append(z_0)
    z.append(z_1)
    n=1
    while abs(z[n-1]-z[n])>params["tol"]:
        z.append((z[n-1]*F_2(z[n],params)-z[n]*F_2(z[n-1],params))/(F_2(z[n],params)-F_2(z[n-1],params)))
        n+=1
    return z[n],n

def shoot_1(s: float, params: dict[str, float]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """solves the first ivp for a guessed initial slope s.
    
    Args:
    s: the guessed initial slope
    params: a dictionary of given parameters

    Returns:
    x_array: an array of x values
    y_array: an array of y and y' values
    """
    params["y'_0"] = s
    x_list, y_list,_,_,_,_ = IVP_solver_1(params=params)
    y_array = np.stack(y_list)
    x_array = np.array(x_list)
    return x_array, y_array

def F_1( s: float, params: dict[str, float]) -> float:
    """shooting residual: y(xend) - yb.
    
    Args:
    s: the guessed initial slope
    params: a dictionary of given parameters

    Returns:
    error: the error in the y values at the boundary 
    """
    _, Y = shoot_1(s, params)
    return Y[-1, 0] - 0

def shooting_method_1(params: dict[str, float], s_init: npt.NDArray[np.float64],shooting_tol:float) -> tuple[object, object, object]:
    """runs the shooting method for the first boundary value problem.
    
    Args:
    params: a dictionary of given parameters
    s_init: an array of the two initial slope guesses
    shooting_tol: the tolerance for the shooting iteration

    Returns:
    s_list: a list of slope guesses from all iterations
    x_arrays: a list of x arrays from all iterations
    y_arrays: a list of y arrays from all iterations
    """
    x_arrays = List.empty_list(types.float64)
    y_arrays = List.empty_list(types.float64)
    s_list = List.empty_list(types.float64)
    s_list.append(s_init[0])
    s_list.append(s_init[1])
    for s in s_init:
        x_array, y_array = shoot_1(s, params)
        x_arrays.append(x_array)
        y_arrays.append(y_array)
    n=1
    while abs(s_list[n]-s_list[n-1]) >= shooting_tol:
        s_new,_ = sekant_root_1(s_list[n-1],s_list[n],params)
        s_list.append(s_new)
        x_array, y_array = shoot_1(s_new, params)
        x_arrays.append(x_array)
        y_arrays.append(y_array)
        n+=1
    return s_list, x_arrays, y_arrays

def shoot_2(s: float, params: dict[str, float]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """solves the second ivp for a guessed initial slope s.
    
    Args:
    s: the guessed initial slope
    params: a dictionary of given parameters

    Returns:
    x_array: an array of x values
    y_array: an array of y and y' values
    """
    params["y'_0"] = s
    x_list, y_list,_,_,_,_ = IVP_solver_2(params)
    y_array = np.stack(y_list)
    x_array = np.array(x_list)
    return x_array, y_array

def F_2( s: float, params) -> float:
    """shooting residual: y(xend) - yb.
    
    Args:
    s: the guessed initial slope
    params: a dictionary of given parameters

    Returns:
    error: the error in the y values at the boundary 
    """
    _, Y = shoot_2(s, params)
    return Y[-1, 0]

def shooting_method_2(params, s_init: npt.NDArray[np.float64],shooting_tol) -> tuple[object, object, object]:
    """runs the shooting method for the second boundary value problem.
    
    Args:
    params: a dictionary of given parameters
    s_init: an array of the two initial slope guesses
    shooting_tol: the tolerance for the shooting iteration

    Returns:
    s_list: a list of slope guesses from all iterations
    x_arrays: a list of x arrays from all iterations
    y_arrays: a list of y arrays from all iterations
    """
    x_arrays = List.empty_list(types.float64)
    y_arrays = List.empty_list(types.float64)
    s_list = List.empty_list(types.float64)
    s_list.append(s_init[0])
    s_list.append(s_init[1])
    for s in s_init:
        x_array, y_array = shoot_2(s, params)
        x_arrays.append(x_array)
        y_arrays.append(y_array)
    n=1
    while abs(s_list[n]-s_list[n-1]) >= shooting_tol:
        s_new,_ = sekant_root_2(s_list[n-1],s_list[n],params)
        s_list.append(s_new)
        x_array, y_array = shoot_2(s_new, params)
        x_arrays.append(x_array)
        y_arrays.append(y_array)
        n+=1
    return s_list, x_arrays, y_arrays
