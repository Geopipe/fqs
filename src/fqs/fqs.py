# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 13:50:23 2019

@author: NKrvavica
"""

import sys
import traceback
from typing import Tuple, Union

import tensorflow as tf


def _filtered_tb():
    """
    Returns a multiline string containing a traceback,
    filtering out noise from Tensorflow's AutoGraph process.

    Useful in conjunction with _tf_print_complex
    """
    return "\n%s" % ("\n".join(line.strip() for line in traceback.format_stack()
                               if ("ag__.converted_call" in line) and not ("_filtered_tb" in line)))


def _tf_print_complex(*args):
    """
    similar to tf.print, but unpack complex tensors into real + imag components to print separately,
    as tf.print only prints `?` for complex tensors.
    """
    return tf.print(tf.get_current_name_scope(),
                    *[(tf.math.real(arg),
                       tf.math.imag(arg)) if isinstance(arg, tf.Tensor) and arg.dtype.is_complex else arg
                      for arg in args],
                    output_stream=sys.stderr)


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.NAME_SCOPES)
def ensure_complex(z: tf.Tensor) -> tf.Tensor:
    """
    No-op if z is already complex. Cast to corresponding complex type if z is float32 or float64.
    """
    with tf.name_scope("EnsureComplex"):
        if z.dtype.is_complex:
            return z
        elif z.dtype is tf.float32:
            return tf.cast(z, tf.complex64)
        elif z.dtype is tf.float64:
            return tf.cast(z, tf.complex128)
        else:
            raise TypeError("z must be already complex, float32, or float64")


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.NAME_SCOPES)
def _safe_denom(d: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Returns a pair in which the first element is a mask with `1` entries where `d` was `0`,
    and the second element is a copy of `d` where `0` has been replaced with `1`.
    """
    with tf.name_scope("SafeDenominator"):
        flag_degenerate = (d == tf.cast(0, dtype=d.dtype))
        return flag_degenerate, tf.where(flag_degenerate, tf.cast(1, dtype=d.dtype), d)


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.NAME_SCOPES)
@tf.custom_gradient
def square_root(p):
    ''' Compute square root of a number with a stable gradient'''
    with tf.name_scope("square_root"):
        root = tf.sqrt(p)

        # implement gradient clipping because d/dx x^(1/2) = (1/2) * x^(-1/2) which has a singularity
        @tf.function
        def grad(upstream):
            with tf.name_scope("grad_of_square_root"):
                droot_dp = (0.5 / (root + 1e-4))
                grad_root = droot_dp * upstream
                return grad_root

        return root, grad


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.NAME_SCOPES)
@tf.custom_gradient
def cubic_root(p):
    ''' Compute cubic root of a number while maintaining its sign'''
    with tf.name_scope("cubic_root"):
        third = 1. / 3.
        real_part_nonnegative = tf.math.real(p) >= 0
        signs = tf.cast(tf.where(real_part_nonnegative, 1.0, -1.0), p.dtype)
        p = signs * p

        ### Mitigation of https://github.com/tensorflow/tensorflow/issues/60468
        degenerate_p, p = _safe_denom(p)
        unsigned_root = tf.where(degenerate_p, tf.constant(0., p.dtype), tf.exp(third * tf.math.log(p)))

        root = signs * unsigned_root

        # implement gradient clipping because d/dx x^(1/3) = (1/3) * x^(-2/3) which has a singularity
        @tf.function
        def grad(upstream):
            with tf.name_scope("grad_of_cubic_root"):
                unsigned_root_plus_eps = unsigned_root + 1e-4
                denom = 3 * unsigned_root_plus_eps * unsigned_root_plus_eps + 1e-4
                droot_dp = (1. / denom)
                grad_root = droot_dp * upstream
                return grad_root

        return root, grad


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.NAME_SCOPES)
def linear(a0: tf.Tensor, b0: tf.Tensor, **kwargs) -> tf.Tensor:
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
    r1: Output data is a scalar tensor containing root of a given polynomial.
    '''
    with tf.name_scope("linear"):
        # n.b. this will give nonsense solutions and really should return NaN at a0 because there is no solution
        (_, a0) = _safe_denom(a0)
        return -b0 / a0


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.NAME_SCOPES)
def quadratic(a0: tf.Tensor,
              b0: tf.Tensor,
              c0: tf.Tensor,
              assume_quadratic=True,
              **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
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
    with tf.name_scope("quadratic"):
        if not assume_quadratic:
            (flag_degenerate, a0) = _safe_denom(a0)
        # Reduce the quadratic equation to the form:
        #    x^2 + bx + c = 0
        b, c = b0 / a0, c0 / a0

        # Some repeating variables
        n_half_b = -0.5 * b
        delta = n_half_b * n_half_b - c
        sqrt_delta = square_root(ensure_complex(delta))
        n_half_b = ensure_complex(n_half_b)

        # Roots
        r1 = n_half_b - sqrt_delta
        r2 = n_half_b + sqrt_delta

        if assume_quadratic:
            return r1, r2
        else:
            # linear fall-back
            with tf.name_scope("linear_fallback"):
                flag_degenerate = (a0 == 0)
                r0 = ensure_complex(linear(b0, c0))
                return tf.where(flag_degenerate, r0, r1), tf.where(flag_degenerate, r0, r2)


@tf.function(experimental_autograph_options=tf.autograph.experimental.Feature.NAME_SCOPES)
def cubic(a0: tf.Tensor,
          b0: tf.Tensor,
          c0: tf.Tensor,
          d0: tf.Tensor,
          first_root_only: bool = False,
          assume_cubic: bool = True,
          **kwargs) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]:
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
    with tf.name_scope("cubic"):
        if not assume_cubic:
            (quadratic_mask, a0) = _safe_denom(a0)
        #if unique_real_root and (a0.dtype.is_complex or b0.dtype.is_complex or c0.dtype.is_complex or d0.dtype.is_complex):
        #    raise TypeError("There may be no real root if the coefficients are complex")

        # Reduce the cubic equation to the form:
        #    x^3 + b*x^2 + c*x + d = 0
        b, c, d = b0 / a0, c0 / a0, d0 / a0

        # Some repeating constants and variables
        third = 1. / 3.
        neg_third_b = -b * third
        sqr3 = 3.**0.5

        # a = 1 so dropped from all remaining formulas
        b_squared = b * b
        delta0 = b_squared - 3 * c
        delta1 = b * (2 * b_squared - 9 * c) + 27 * d

        null_delta0 = (delta0 == 0)
        null_delta1 = (delta1 == 0)

        def single_repeated_root(**kwargs):
            with tf.name_scope("triple_root"):
                r1 = neg_third_b
                r1 = ensure_complex(r1)
                if first_root_only:
                    return r1,
                else:
                    return r1, r1, r1

        # general_case
        def multiple_roots(**kwargs):
            nonlocal delta0, delta1  # we assign to them now, so can't capture implicitly
            with tf.name_scope("multiple_roots"):
                # pull out a bunch of intermediate terms for debugging / steppability
                deltas_sqrt_arg = delta1 * delta1 - 4. * delta0 * delta0 * delta0
                deltas_sqrt_arg = ensure_complex(deltas_sqrt_arg)
                delta0 = ensure_complex(delta0)
                delta1 = ensure_complex(delta1)
                C1_cubed = (delta1 + square_root(deltas_sqrt_arg)) / 2.
                C1 = cubic_root(C1_cubed)
                # C1 = cubic_root((_ensure_complex(delta1) + square_root(_ensure_complex(delta1 * delta1 - 4. * delta0 * delta0 * delta0))) / 2.)
                # algebraically, C1==0 precisely when delta0==0.
                # numerically, if this turns out to be a problem,
                # can always add a second test for explicit C1 == 0
                C_branches = (C1, (-1. + sqr3 * 1j) * C1 / 2, (-1. - sqr3 * 1j) * C1 / 2)
                r_branch = lambda i: ensure_complex(neg_third_b) - third * (C_branches[i] + delta0 / tf.where(
                    null_delta0, tf.cast(1. + 0j, C1.dtype), C_branches[i]))
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
        root_branches = zip(single_repeated_root(**kwargs), multiple_roots(**kwargs))
        cubic_roots = tuple(
            tf.where(triple_root_mask, single_branch, multiple_branch)
            for single_branch, multiple_branch in root_branches)
        if not assume_cubic:

            def quadratic_roots(**kwargs):
                with tf.name_scope("quadratic_fallback"):
                    r1, r2 = quadratic(b0, c0, d0, assume_quadratic=False, **kwargs)
                    if first_root_only:
                        return r1
                    else:
                        return r1, r1, r2

            root_branches = zip(quadratic_roots(**kwargs), cubic_roots)
            cubic_roots = tuple(
                tf.where(quadratic_mask, quadratic_branch, cubic_branch)
                for quadratic_branch, cubic_branch in root_branches)

        return cubic_roots


@tf.function
def quartic(a0: tf.Tensor,
            b0: tf.Tensor,
            c0: tf.Tensor,
            d0: tf.Tensor,
            e0: tf.Tensor,
            assume_quartic: bool = True,
            **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
    with tf.name_scope("quartic"):
        if not assume_quartic:
            (cubic_mask, a0) = _safe_denom(a0)
        # Reduce the quartic equation to the form:
        #    x^4 + b*x^3 + c*x^2 + d*x + e = 0
        with tf.name_scope("monic_reduction"):
            b, c, d, e = b0 / a0, c0 / a0, d0 / a0, e0 / a0

        with tf.name_scope("intermediate_cubic_coefficients"):
            # Some repeating variables
            quarter_b = 0.25 * b
            sixteenth_b2 = quarter_b * quarter_b

            # Coefficients of subsidiary cubic equation
            p = 3 * sixteenth_b2 - 0.5 * c
            q = b * sixteenth_b2 - c * quarter_b + 0.5 * d
            r = 3 * sixteenth_b2 * sixteenth_b2 - c * sixteenth_b2 + d * quarter_b - e

        with tf.name_scope("intermediate_cubic_root"):
            # One root of the cubic equation
            half_q_sq = 0.5 * q * q
            z0, = cubic(1, p, r, p * r - half_q_sq, first_root_only=True, **kwargs)

        with tf.name_scope("intermediate_quadratic_coefficients"):
            # Additional variables
            s = square_root(ensure_complex(2. * p) + tf.cast(2, z0.dtype) * z0)
            null_s, s_denom = _safe_denom(s)  # s gets reused, so don't overwrite it
            t = tf.where(null_s, z0 * z0 + ensure_complex(r), ensure_complex(-q) / s_denom)

            # Compute roots by quadratic equations
            one_complex = tf.constant(1, dtype=s.dtype)

        with tf.name_scope("intermediate_quadratic_roots"):
            r0, r1 = quadratic(one_complex, s, ensure_complex(z0 + t), **kwargs)
            r2, r3 = quadratic(one_complex, -s, ensure_complex(z0 - t), **kwargs)

        with tf.name_scope("final_quartic_roots"):
            quarter_b = ensure_complex(quarter_b)
            quartic_roots = (r0 - quarter_b, r1 - quarter_b, r2 - quarter_b, r3 - quarter_b)

        if assume_quartic:
            return quartic_roots
        else:

            def cubic_roots(**kwargs):
                with tf.name_scope("cubic_fallback"):
                    r1, r2, r3 = cubic(b0, c0, d0, e0, assume_cubic=False, **kwargs)
                    return (r1, r1, r2, r3)

            root_branches = zip(cubic_roots(**kwargs), quartic_roots)
            return tuple(
                tf.where(cubic_mask, cubic_branch, quartic_branch) for cubic_branch, quartic_branch in root_branches)
