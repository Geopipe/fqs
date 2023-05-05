#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Authors: elfprince13 and tnybny, based on original library by NKrvavica
MIT Licensed
"""

import numpy as np
try:
    # if run as module
    from fqs.fqs import quartic
    from fqs.orthogonal_least_quartics import coefs
except ModuleNotFoundError:
    from fqs import quartic
    from orthogonal_least_quartics import coefs
import tensorflow as tf
from tensorflow.experimental.numpy import isclose as tf_isclose

def _retain_real_roots(roots):
    xs = tf.math.real(roots)
    ys = tf.math.imag(roots)
    return tf.gather(xs, tf.where(tf_isclose(ys, 0., atol=1e-06)))

def _run_one_test(x, y):
    a, b, c, d, e = coefs(x, y)
    roots = quartic(a, b, c, d, e, assume_quartic=False) 
    roots = _retain_real_roots(roots)
    return roots
    
class RegressingAngleTest(tf.test.TestCase):

    def test_x_eq_0(self) -> None:
        x = tf.constant([0, 0, 0], dtype=tf.float32)
        y = tf.constant([-1, 0, 1], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, tf_isclose(roots, 0.))

    def test_y_eq_0(self) -> None:
        x = tf.constant([-1, 0, 1], dtype=tf.float32)
        y = tf.constant([0, 0, 0], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, tf_isclose(roots, 0.))

    def test_x_eq_y(self) -> None:
        x = tf.constant([-1, 0, 1], dtype=tf.float32)
        y = tf.constant([-1, 0, 1], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, tf.logical_or(tf_isclose(roots, -1.), tf_isclose(roots, 1.)))

    def test_x_eq_ny(self) -> None:
        x = tf.constant([-1, 0, 1], dtype=tf.float32)
        y = tf.constant([1, 0, -1], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, tf.logical_or(tf_isclose(roots, -1.), tf_isclose(roots, 1.)))

    def test_diamond(self) -> None:
        x = tf.constant([-2, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2], dtype=tf.float32)
        y = tf.constant([0, -1, 0, 1, -2, -1, 0, 1, 2, -1, 0, 1, 0], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, np.logical_or(tf_isclose(roots, -1.), tf_isclose(roots, 1.)))

    def test_vert_rect(self) -> None:
        x = tf.constant([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], dtype=tf.float32)
        y = tf.constant([-1, -1, 0, 0, 1, 1], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, tf_isclose(roots, 0., atol=1e-6))

    def test_horiz_rect(self) -> None:
        x = tf.constant([-1, -1, 0, 0, 1, 1], dtype=tf.float32)
        y = tf.constant([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, tf_isclose(roots, 0., atol=1e-6))

    def test_square(self) -> None:
        x = tf.constant([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=tf.float32)
        y = tf.constant([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=tf.float32)
        roots = _run_one_test(x, y)
        self.assertIn(True, tf_isclose(roots, 0., atol=1e-6))

if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    tf.test.main()
