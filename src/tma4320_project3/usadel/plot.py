import numpy as np
import numpy.typing as npt
from typing import cast
from collections.abc import Callable

from .solver import(
    make_diff_system,
    make_bc
)

from .observables import(
    dos,
    current_integrand
)

