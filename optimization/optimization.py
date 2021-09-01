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
import time
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

def chi2(p: ArrayLike, 
            x: ArrayLike, 
            y: ArrayLike,
            w: ArrayLike,
            model: Callable)->ArrayLike:
    """Chi2.

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
    res = np.sum(np.absolute(residuals(p, x, y, w, model))**2)
    return res

def lm_func(p: ArrayLike, 
            x: ArrayLike, 
            y: ArrayLike,
            w: ArrayLike,
            model: Callable)->ArrayLike:
    """Absolute residuals for Levenberg-Marquardt optimizer.

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


def nm_func(p: ArrayLike, 
            x: ArrayLike, 
            y: ArrayLike,
            w: ArrayLike,
            model: Callable)->ArrayLike:
    """Chi2 for simplex.

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
    res = chi2(p, x, y, w, model)
    return res

x = np.linspace(0, 10, 10000)
p0 = np.asarray((1.1, 2.3))
y = model(p0, x)
w = np.ones_like(y)

p0 = np.asarray((10, 10))
t0 = time.time()
popt, cov, infodict, ier, msg = optimize.leastsq(lm_func, p0, args=(x, y, w, model), full_output=True)
t1 = time.time()
t = (t1-t0)*1e3
print('optimize.leastsq: ', popt, t)


p0 = np.asarray((10, 10))
t0 = time.time()
popt = optimize.fmin(nm_func, p0, args=(x, y, w, model), disp=0)
t1 = time.time()
t = (t1-t0)*1e3
print('optimize.fmin: ', popt, t)


p0 = np.asarray((10, 10))
t0 = time.time()
res = optimize.least_squares(lm_func, p0, args=(x, y, w, model))
t1 = time.time()
t = (t1-t0)*1e3
print('optimize.least_square (trf): ', res.x, t)

p0 = np.asarray((10, 10))
t0 = time.time()
res = optimize.least_squares(lm_func, p0, args=(x, y, w, model), method='lm')
t1 = time.time()
t = (t1-t0)*1e3
print('optimize.least_square (lm): ', res.x, t)
print(chi2(res.x, x, y, w, model), res.cost*2)


p0 = np.asarray((10, 10))
t0 = time.time()
res = optimize.minimize(nm_func, p0, args=(x, y, w, model), method='nelder-mead')
t1 = time.time()
t = (t1-t0)*1e3
print('optimize.minimize (nerlder): ', res.x, t)
print(chi2(res.x, x, y, w, model), res.fun)