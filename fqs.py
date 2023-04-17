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

def linear(a0 : tf.Tensor, b0 : tf.Tensor) -> tf.Tensor:
        ''' Analytical solver for a single linear equation
    (1st order polynomial).

    Parameters
    ----------
    a0, b0: array_like
        Input data are coefficients of the Linear polynomial::

            a0*x + b0 = 0

    the `a0` elements must all be non-zero

    Returns
    -------
    r1, r2: tuple
        Output data is a tuple of two roots of a given polynomial.
    '''
        return -b0 / a0

def quadratic(a0 : tf.Tensor, b0 : tf.Tensor, c0 : tf.Tensor, assume_quadratic = True) -> Tuple[tf.Tensor, tf.Tensor]:
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
    #    x^2 + bx + c = 0
    b, c = b0 / a0, c0 / a0

    # Some repeating variables
    n_half_b = -0.5*b
    delta = n_half_b*n_half_b - c
    sqrt_delta = tf.sqrt(_ensure_complex(delta))
    n_half_b = _ensure_complex(n_half_b)

    # Roots
    r1 = n_half_b - sqrt_delta
    r2 = n_half_b + sqrt_delta

    if assume_quadratic:
        return r1, r2
    else:
        # linear fall-back
        flag_degenerate = (a0 == 0)
        r0 = _ensure_complex(linear(b0, c0))
        return tf.where(flag_degenerate, r0, r1), tf.where(flag_degenerate, r0, r2)

def cubic(a0 : tf.Tensor, b0 : tf.Tensor, c0 : tf.Tensor, d0 : tf.Tensor, first_root_only : bool = False, assume_cubic : bool = True) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]:
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

    #if unique_real_root and (a0.dtype.is_complex or b0.dtype.is_complex or c0.dtype.is_complex or d0.dtype.is_complex):
    #    raise TypeError("There may be no real root if the coefficients are complex")

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
        r1 = _ensure_complex(r1)
        if first_root_only:
            return r1,
        else:
            return r1, r1, r1
    
    # general_case
    def multiple_roots():
        C1 = cubic_root((_ensure_complex(delta1) + tf.sqrt(_ensure_complex(delta1 * delta1 - 4. * delta0 * delta0 * delta0))) / 2.)
        C_branches = (C1, (-1. + sqr3 * 1j) * C1 / 2, (-1. - sqr3 * 1j) * C1 / 2)
        r_branch = lambda i: _ensure_complex(neg_third_b) - third * (C_branches[i] + _ensure_complex(delta0) / C_branches[i])
        r1 = r_branch(0)
        if first_root_only:
            # this is only valid if the tests for complex coefficients has run above.
            return r1,
        else:
            r2 = r_branch(1)
            r3 = r_branch(2)
            return r1, r2, r3
        
    
    # Masks for different combinations of roots
    triple_root_mask = null_delta0 & null_delta1
    root_branches = zip(single_repeated_root(), multiple_roots())
    cubic_roots = tuple(tf.where(triple_root_mask, single_branch, multiple_branch) for single_branch, multiple_branch in root_branches)
    if assume_cubic:
        return cubic_roots
    else:
        def quadratic_roots():
            r1, r2 = quadratic(b0, c0, d0, assume_quadratic=False)
            if first_root_only:
                return r1
            else:
                return r1, r1, r2
        quadratic_mask = (a0 == 0)
        root_branches = zip(quadratic_roots(), cubic_roots)
        return tuple(tf.where(quadratic_mask, quadratic_branch, cubic_branch) for quadratic_branch, cubic_branch in root_branches)

    

def quartic(a0 : tf.Tensor, b0 : tf.Tensor, c0 : tf.Tensor, d0 : tf.Tensor, e0 : tf.Tensor, assume_quartic : bool = True) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
    #    x^4 + b*x^3 + c*x^2 + d*x + e = 0
    b, c, d, e = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    quarter_b = 0.25*b
    sixteenth_b2 = quarter_b*quarter_b

    # Coefficients of subsidiary cubic euqtion
    p = 3*sixteenth_b2 - 0.5*c
    q = b*sixteenth_b2 - c*quarter_b + 0.5*d
    r = 3*sixteenth_b2*sixteenth_b2 - c*sixteenth_b2 + d*quarter_b - e

    # One root of the cubic equation
    z0, = cubic(1, p, r, p*r - 0.5*q*q, first_root_only = True)

    # Additional variables
    s = tf.sqrt(_ensure_complex(2*p) + 2*z0)
    t = tf.where(s == 0, z0*z0 + _ensure_complex(r), _ensure_complex(-q) / s)

    # Compute roots by quadratic equations
    one_complex = tf.constant(1, dtype=s.dtype)
    r0, r1 = quadratic(one_complex, s, _ensure_complex(z0 + t))
    r2, r3 = quadratic(one_complex, -s, _ensure_complex(z0 - t))

    quarter_b = _ensure_complex(quarter_b)

    quartic_roots = (r0 - quarter_b, r1 - quarter_b, r2 - quarter_b, r3 - quarter_b)
    if assume_quartic:
        return quartic_roots
    else:
        def cubic_roots():
            r1, r2, r3 = cubic(b0, c0, d0, e0, assume_cubic=False)
            return (r1, r1, r2, r3)
        cubic_mask = (a0 == 0)
        root_branches = zip(cubic_roots(), quartic_roots)
        return tuple(tf.where(cubic_mask, cubic_branch, quartic_branch) for cubic_branch, quartic_branch in root_branches)