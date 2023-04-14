# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 13:50:23 2019

@author: NKrvavica
"""

import tensorflow as tf

from typing import Union, Tuple

def _ensure_complex(z : tf.Tensor) -> tf.Tensor:
    if z.dtype.is_complex:
        return z
    elif z.dtype is tf.float32:
        return tf.cast(z, tf.complex64)
    elif z.dtype is tf.float64:
        return tf.cast(z, tf.complex128)
    else:
        raise TypeError("z must be already complex, float32, or float64")

def quadratic(a0 : tf.Tensor, b0 : tf.Tensor, c0 : tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    ''' Analytical solver for a single quadratic equation
    (2nd order polynomial).

    Parameters
    ----------
    a0, b0, c0: array_like
        Input data are coefficients of the Quadratic polynomial::

            a0*x^2 + b0*x + c0 = 0

    the `a0` elements must all be non-zero

    Returns
    -------
    r1, r2: tuple
        Output data is a tuple of two roots of a given polynomial.
    '''
    # Reduce the quadratic equation to the form:
    #    x^2 + ax + b = 0
    a, b = b0 / a0, c0 / a0

    # Some repating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = tf.sqrt(_ensure_complex(delta))
    a0 = _ensure_complex(a0)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2

def cubic(a0 : tf.Tensor, b0 : tf.Tensor, c0 : tf.Tensor, d0 : tf.Tensor, unique_real_root : bool = False) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]:
    ''' Analytical closed-form solver for a single cubic equation
    (3rd order polynomial), gives all three roots.

    Parameters
    ----------
    a0, b0, c0, d0: array_like
        Input data are coefficients of the Cubic polynomial::

            a0*x^3 + b0*x^2 + c0*x + d0 = 0

    the `a0` elements must all be non-zero

    Returns
    -------
    roots: tuple
        Output data is a tuple of three roots of a given polynomial.
    '''

    if unique_real_root and (a0.dtype.is_complex or b0.dtype.is_complex or c0.dtype.is_complex or d0.dtype.is_complex):
        raise TypeError("There may be no real root if the coefficients are complex")

    # Reduce the cubic equation to the form:
    #    x^3 + b*x^2 + c*x + d = 0
    b, c, d = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    neg_third_b = -b*third    
    sqr3 = 3.**0.5

    # a = 1 so dropped from all remaining formulas
    b_squared = b * b
    delta0 = b_squared - 3 * c
    delta1 = b * (2 * b_squared - 9 * c) + 27 * d

    null_delta0 = delta0 == 0
    null_delta1 = delta1 == 0

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign'''
        x_real_nonneg = tf.math.real(x) >= 0
        return tf.where(x_real_nonneg, x**third, -(-x)**third)
            
    def single_repeated_root():
        r1 = neg_third_b 
        if unique_real_root:
            return r1,
        else:
            r1 = _ensure_complex(r1)
            return r1, r1, r1
    
    # general_case
    def multiple_roots():
        sqr_branches = _ensure_complex(tf.where(null_delta0, tf.constant(1., dtype=delta0.dtype), tf.constant(-1., dtype=delta0.dtype)))
        C1 = cubic_root((_ensure_complex(delta1) + sqr_branches * tf.sqrt(_ensure_complex(delta1 * delta1 - 4. * delta0 * delta0 * delta0))) / 2.)
        C_branches = (C1, (-1. + sqr3 * 1j) * C1 / 2, (-1. - sqr3 * 1j) * C1 / 2)
        rs = tuple(_ensure_complex(neg_third_b) - third * (C + _ensure_complex(delta0) / C) for C in C_branches)
        if unique_real_root:
            # this is only valid if the tests for complex coefficients has run above.
            return tf.math.real(rs[0]),
        else:
            return rs
    
    # Masks for different combinations of roots
    triple_root_mask = null_delta0 & null_delta1
    root_branches = zip(single_repeated_root(), multiple_roots())
    return tuple(tf.where(triple_root_mask, single_branch, multiple_branch) for single_branch, multiple_branch in root_branches)
    

def quartic(a0 : tf.Tensor, b0 : tf.Tensor, c0 : tf.Tensor, d0 : tf.Tensor, e0 : tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    ''' Analytical closed-form solver for a single quartic equation
    (4th order polynomial). Calls `cubic` and
    `quadratic`.

    Parameters
    ----------
    a0, b0, c0, d0, e0: array_like
        Input data are coefficients of the Quartic polynomial::

        a0*x^4 + b0*x^3 + c0*x^2 + d0*x + e0 = 0

    the `a0` elements must all be non-zero

    Returns
    -------
    r1, r2, r3, r4: tuple
        Output data is a tuple of four roots of given polynomial.
    '''

    # Reduce the quartic equation to the form:
    #    x^4 + a*x^3 + b*x^2 + c*x + d = 0
    a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    a0 = 0.25*a
    a02 = a0*a0

    # Coefficients of subsidiary cubic euqtion
    p = 3*a02 - 0.5*b
    q = a*a02 - b*a0 + 0.5*c
    r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    z0 = cubic(1, p, r, p*r - 0.5*q*q, unique_real_root = True)

    # Additional variables
    s = tf.sqrt(_ensure_complex(2*p + 2*z0))
    if s == 0:
        t = _ensure_complex(z0*z0 + r)
    else:
        t = _ensure_complex(-q) / s

    # Compute roots by quadratic equations
    one_complex = tf.constant(1, dtype=s.dtype)
    r0, r1 = quadratic(one_complex, s, _ensure_complex(z0 + t))
    r2, r3 = quadratic(one_complex, -s, _ensure_complex(z0 - t))

    a0 = _ensure_complex(a0)

    return r0 - a0, r1 - a0, r2 - a0, r3 - a0