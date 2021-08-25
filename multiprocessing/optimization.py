"""
Optimization Examples.

Copyright (C) 2021-2022 Milan Skocic

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author(s): Milan Skocic <milan.skocic@gmail.com>
"""
from tkinter.constants import W
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize

def model(p: ArrayLike, x: ArrayLike)->ArrayLike:
    """Model to compute values to be compared with experimental data.

    Parameters
    ----------
    p : ArrayLike
        Parameters array
    x : ArrayLike
        Dependant values array

    Returns
    -------
    y: ArrayLike
        Computed values.
    """
    y = p[0] + p[1] * x

    return y

def residuals(p: ArrayLike, 
              x: ArrayLike, 
              y: ArrayLike,
              w: ArrayLike,
              model: Callable)->ArrayLike:
    """Algebric residuals.

    Parameters
    ----------
    p : ArrayLike
        Parameters
    x : ArrayLike
        Dependant variable
    y : ArrayLike
        Experimental independant data
    w : ArrayLike
        Weights
    model : Callable
        Model to be used

    Returns
    -------
    res: ArrayLike
        Algebric residuals.
    """
    res = (model(p, x) - y) * w
    return res

def lm_func(p: ArrayLike, 
            x: ArrayLike, 
            y: ArrayLike,
            w: ArrayLike,
            model: Callable)->ArrayLike:
    """Absolute residuals

    Parameters
    ----------
    p : ArrayLike
        Parameters
    x : ArrayLike
        Dependant variable
    y : ArrayLike
        Experimental independant data
    w : ArrayLike
        Weights
    model : Callable
        Model to be used

    Returns
    -------
    res: ArrayLike
        Algebric residuals.
    """
    res = np.absolute(residuals(p, x, y, w, model))
    return res