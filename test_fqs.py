import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly
import tensorflow as tf
from typing import Any, Callable, FrozenSet, Set
from fqs import quadratic, cubic, quartic

degrees = {quadratic : 2, cubic : 3, quartic : 4}

def random_roots(solver : Callable, samples : int = 3, lb : float = -3, ub : float = 3, complex : bool = False) -> np.ndarray:
    sampler = lambda: np.random.uniform(lb, ub, (samples,degrees[solver]))
    real = sampler()
    if complex:
        return real + sampler() * 1j
    else:
        return real


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

def assertSetsClose(test : tf.test.TestCase, expected : tf.Tensor, actual : tf.Tensor, precision : float = 1.e6):
    expected = tf.cast(round_complex(tf.constant(expected * precision)), dtype=expected.dtype) / precision
    actual = tf.cast(round_complex(tf.constant(actual * precision)), dtype=actual.dtype) / precision
    test.assertSetEqual(tensor_2d_to_nested_sets(expected), tensor_2d_to_nested_sets(actual))

class QuadraticTest(tf.test.TestCase):
    def test_identical_real_roots(self):
        expected = np.array([[i]*2 for i in range(3)], dtype=np.float32)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(quadratic)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_identical_complex_roots(self):
        expected = np.array([[i-(i*1j)]*2 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(quadratic, complex=True)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

class CubicTest(tf.test.TestCase):
    def test_identical_real_roots(self):
        expected = np.array([[i]*3 for i in range(3)], dtype=np.float32)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_real_roots(self):
        expected = np.array([[i]+[i+1]*2 for i in range(0,6,2)], dtype=np.float32)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(cubic)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)
    
    def test_identical_complex_roots(self):
        expected = np.array([[i+((i+1)*1j)]*3 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_complex_roots(self):
        expected = np.array([[-i+((i+1)*1j)]+[i-((i+1)*1j)]*2 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(cubic, complex = True)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)



class QuarticTest(tf.test.TestCase):
    def test_identical_real_roots(self):
        expected = np.array([[i]*4 for i in range(3)], dtype=np.float32)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_three_identical_real_roots(self):
        expected = np.array([[i]+[i+1]*3 for i in range(0,6,2)], dtype=np.float32)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_biquadratic_real_roots(self):
        expected = np.array([[i]*2+[i+1]*2 for i in range(0,6,2)], dtype=np.float32)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(quartic)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)
    
    def test_identical_complex_roots(self):
        expected = np.array([[i+((i+1)*1j)]*4 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_three_identical_complex_roots(self):
        expected = np.array([[-i+((i+1)*1j)]+[i-((i+1)*1j)]*3 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_biquadratic_complex_roots(self):
        expected = np.array([[-i+((i+1)*1j)]*2+[i-((i+1)*1j)]*2 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(quartic, complex=True)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)



if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    tf.test.main()