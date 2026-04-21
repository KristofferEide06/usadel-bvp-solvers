import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

def plot_solution(
        ax: plt.Axes, 
        X: npt.NDArray[np.float64], 
        Y: npt.NDArray[np.float64], 
        label: str = "",
        xlabel: str="x", 
        ylabel: str ="y", 
        **plot_kwargs,
        ) -> None:
    """plot a single solution on the given axes.
    
    Args:
        ax: axis to plot on
        X: array of x axis values
        Y: array of y axis values
        label: label of graf
        xlabel: label of x axis
        ylabel: label of y axis

    Returns:
        None: displays plot
    """

    ax.plot(X, Y, label=label, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if label:
        ax.legend()