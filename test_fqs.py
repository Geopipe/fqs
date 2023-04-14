import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly
import tensorflow as tf
from typing import Any, FrozenSet, Set
from fqs import quadratic, cubic, quartic

degrees = {quadratic : 2, cubic : 3, quartic : 4}

"""
These tests are sensitive to the order the roots are given in!
"""


def _run_one_test(rootses, solver):
    coefs = np.apply_along_axis(lambda roots: Poly.fromroots(roots).coef, -1, rootses)
    # np.polynomial takes coefs in increase order but fqs takes them in decreasing order.
    return tf.concat(solver(*[tf.constant(coefs[:,i:(i+1)]) for i in reversed(range(1+degrees[solver]))]), axis=-1)

def tensor_2d_to_nested_sets(t : tf.Tensor) -> Set[FrozenSet[Any]]:
    return {frozenset(row) for row in t.numpy()}

def round_complex(t : tf.Tensor) -> tf.Tensor:
    if t.dtype.is_complex:
        return tf.dtypes.complex(tf.round(tf.math.real(t)), tf.round(tf.math.imag(t)))
    else:
        return tf.round(t)

def assertSetsClose(test : tf.test.TestCase, expected : tf.Tensor, actual : tf.Tensor, precision : float = 1e6):
    expected = round_complex(tf.constant(expected * precision))
    actual = round_complex(tf.constant(actual * precision))
    test.assertSetEqual(tensor_2d_to_nested_sets(expected), tensor_2d_to_nested_sets(actual))

class QuadraticTest(tf.test.TestCase):
    def test_identical_real_roots(self):
        expected = np.array([[i]*2 for i in range(3)])
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = np.random.uniform(-3, 3, (3,2))
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_identical_complex_roots(self):
        expected = np.array([[i-(i*1j)]*2 for i in range(3)])
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = np.random.uniform(-3, 3, (3,2)) + np.random.uniform(-3, 3, (3,2)) * 1j
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

class CubicTest(tf.test.TestCase):
    def test_identical_real_roots(self):
        expected = np.array([[i]*3 for i in range(3)])
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_real_roots(self):
        expected = np.array([[i]+[i+1]*2 for i in range(0,6,2)])
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = np.random.uniform(-3, 3, (3,3))
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)
    
    def test_identical_complex_roots(self):
        expected = np.array([[i+((i+1)*1j)]*3 for i in range(3)])
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_complex_roots(self):
        expected = np.array([[-i+((i+1)*1j)]+[i-((i+1)*1j)]*2 for i in range(3)])
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = np.random.uniform(-3, 3, (3,3)) + np.random.uniform(-3, 3, (3,3)) * 1j
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)



if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    tf.test.main()