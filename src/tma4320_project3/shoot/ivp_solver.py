import numpy as np
from numba.typed import List
from numba import njit, types

from .systems import (
    diff_system_1,
    diff_system_2,
)

dict_type = types.DictType(types.unicode_type, types.float64)

@njit
def IVP_solver_1(params: dict[str, float]) -> tuple[object, object, object, object, int, int]:
    """ solves the first given differential system using rk3 and adjusting step length with rk4
    
        Args:
        params: a dictionary of given parameters

        Returns:
        x_list: a list over all the x values
        y_list: a list over arrays containing all y and y' values from rk3
        z_list: a list over arrays containing all y and y' values from rk4
        h_list: a list over all accepted step lengths
        accepted: the amount of accepted steps
        rejected: the amount of rejected steps
    """
    accepted = 0
    rejected = 0
    x_list = List.empty_list(types.float64)
    h_list = List.empty_list(types.float64)
    y_list = List.empty_list(types.float64[:])
    z_list = List.empty_list(types.float64[:])
    x_list.append(params["x_init"])
    y_init = np.empty(2, dtype=np.float64)
    y_init[0] = params["y_0"]
    y_init[1] = params["y'_0"]
    y_list.append(y_init)
    h_list.append(params["h_0"])
    k_1 = diff_system_1(x_list[0],y_list[0])

    while x_list[accepted]<params["x_end"]:
        h_list[accepted] = min(h_list[accepted], params["x_end"]-x_list[accepted])
        k_2 = diff_system_1(x_list[accepted]+1/2*h_list[accepted], y_list[accepted] + 1/2*h_list[accepted]*k_1)
        k_3 = diff_system_1(x_list[accepted] + 3/4*h_list[accepted], y_list[accepted] + 3/4*h_list[accepted]*k_2)
        y_new= y_list[accepted] + 1/9*h_list[accepted]*(2*k_1+3*k_2+4*k_3)
        x_new = x_list[accepted] + h_list[accepted]
        k_4 = diff_system_1(x_new,y_new)
        z_new = y_list[accepted] + 1/24*h_list[accepted]*(7*k_1 + 6*k_2 + 8*k_3 + 3*k_4)
        est = np.sqrt(np.sum((y_new - z_new)**2))
        h_new = (params["alpha"] * h_list[accepted] * (params["tol"]/est)**(1/3))
        if est < params["tol"]:
            accepted+=1
            k_1=k_4
            y_list.append(y_new)
            x_list.append(x_new)
            z_list.append(z_new)
            h_list.append(h_new)
        else: 
            h_list[accepted] = h_new
            rejected +=1
    return x_list, y_list, z_list, h_list, accepted, rejected

@njit
def IVP_solver_2(params: dict[str, float]) -> tuple[object, object, object, object, int, int]:
    """ solves the second given differential system using rk3 and adjusting step length with rk4
    
        Args:
        params: a dictionary of given parameters

        Returns:
        x_list: a list over all the x values
        y_list: a list over arrays containing all y and y' values from rk3
        z_list: a list over arrays containing all y and y' values from rk4
        h_list: a list over all accepted step lengths
        accepted: the amount of accepted steps
        rejected: the amount of rejected steps
    """
    accepted = 0
    rejected = 0
    x_list = List.empty_list(types.float64)
    h_list = List.empty_list(types.float64)
    y_list = List.empty_list(types.float64[:])
    z_list = List.empty_list(types.float64[:])
    x_list.append(params["x_init"])
    y_init = np.empty(2, dtype=np.float64)
    y_init[0] = params["y_0"]
    y_init[1] = params["y'_0"]
    y_list.append(y_init)
    h_list.append(params["h_0"])
    k_1 = diff_system_2(x_list[0],y_list[0])

    while x_list[accepted]<params["x_end"]:
        h_list[accepted] = min(h_list[accepted], params["x_end"]-x_list[accepted])
        k_2 = diff_system_2(x_list[accepted]+1/2*h_list[accepted], y_list[accepted] + 1/2*h_list[accepted]*k_1)
        k_3 = diff_system_2(x_list[accepted] + 3/4*h_list[accepted], y_list[accepted] + 3/4*h_list[accepted]*k_2)
        y_new= y_list[accepted] + 1/9*h_list[accepted]*(2*k_1+3*k_2+4*k_3)
        x_new = x_list[accepted] + h_list[accepted]
        k_4 = diff_system_2(x_new,y_new)
        z_new = y_list[accepted] + 1/24*h_list[accepted]*(7*k_1 + 6*k_2 + 8*k_3 + 3*k_4)
        est = np.sqrt(np.sum((y_new - z_new)**2))
        h_new = (params["alpha"] * h_list[accepted] * (params["tol"]/est)**(1/3))
        if est < params["tol"]:
            accepted+=1
            k_1=k_4
            y_list.append(y_new)
            x_list.append(x_new)
            z_list.append(z_new)
            h_list.append(h_new)
        else: 
            h_list[accepted] = h_new
            rejected +=1
    return x_list, y_list, z_list, h_list, accepted, rejected