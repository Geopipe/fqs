from typing import Any, Callable, FrozenSet, Set

import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly
from .fqs import cubic
from .fqs import cubic_root
from .fqs import ensure_complex
from .fqs import linear
from .fqs import quadratic
from .fqs import quartic
from .fqs import square_root
from .orthogonal_least_quartics import coefs
import tensorflow as tf

degrees = {quadratic: 2, cubic: 3, quartic: 4}


def random_roots(solver: Callable,
                 samples: int = 3,
                 lb: float = -3,
                 ub: float = 3,
                 complex: bool = False) -> np.ndarray:
    sampler = lambda: np.random.uniform(lb, ub, (samples, degrees[solver]))
    real = sampler()
    if complex:
        return real + sampler() * 1j
    else:
        return real


def _run_one_test(rootses, solver):
    coefs = np.apply_along_axis(lambda roots: Poly.fromroots(roots).coef, -1, rootses)
    # np.polynomial takes coefs in increase order but fqs takes them in decreasing order.
    return tf.concat(solver(*[tf.constant(coefs[:, i:(i + 1)]) for i in reversed(range(1 + degrees[solver]))]), axis=-1)


def tensor_2d_to_nested_sets(t: tf.Tensor) -> Set[FrozenSet[Any]]:
    return {frozenset(row) for row in t.numpy()}


def round_complex(t: tf.Tensor) -> tf.Tensor:
    if t.dtype.is_complex:
        return tf.dtypes.complex(tf.round(tf.math.real(t)), tf.round(tf.math.imag(t)))
    else:
        return tf.round(t)


def assertSetsClose(test: tf.test.TestCase, expected: tf.Tensor, actual: tf.Tensor, precision: float = 1.e6):
    expected = tf.cast(round_complex(tf.constant(expected * precision)), dtype=expected.dtype) / precision
    actual = tf.cast(round_complex(tf.constant(actual * precision)), dtype=actual.dtype) / precision
    test.assertSetEqual(tensor_2d_to_nested_sets(expected), tensor_2d_to_nested_sets(actual))


class QuadraticTest(tf.test.TestCase):

    def test_identical_real_roots(self):
        expected = np.array([[i] * 2 for i in range(3)], dtype=np.float32)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(quadratic)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_identical_complex_roots(self):
        expected = np.array([[i - (i * 1j)] * 2 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(quadratic, complex=True)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)


class CubicTest(tf.test.TestCase):

    def test_identical_real_roots(self):
        expected = np.array([[i] * 3 for i in range(3)], dtype=np.float32)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_real_roots(self):
        expected = np.array([[i] + [i + 1] * 2 for i in range(0, 6, 2)], dtype=np.float32)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(cubic)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_identical_complex_roots(self):
        expected = np.array([[i + ((i + 1) * 1j)] * 3 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_complex_roots(self):
        expected = np.array([[-i + ((i + 1) * 1j)] + [i - ((i + 1) * 1j)] * 2 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(cubic, complex=True)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)


class QuarticTest(tf.test.TestCase):

    def test_identical_real_roots(self):
        expected = np.array([[i] * 4 for i in range(3)], dtype=np.float32)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_three_identical_real_roots(self):
        expected = np.array([[i] + [i + 1] * 3 for i in range(0, 6, 2)], dtype=np.float32)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_biquadratic_real_roots(self):
        expected = np.array([[i] * 2 + [i + 1] * 2 for i in range(0, 6, 2)], dtype=np.float32)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(quartic)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_identical_complex_roots(self):
        expected = np.array([[i + ((i + 1) * 1j)] * 4 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_three_identical_complex_roots(self):
        expected = np.array([[-i + ((i + 1) * 1j)] + [i - ((i + 1) * 1j)] * 3 for i in range(3)], dtype=np.complex64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_biquadratic_complex_roots(self):
        expected = np.array([[-i + ((i + 1) * 1j)] * 2 + [i - ((i + 1) * 1j)] * 2 for i in range(3)],
                            dtype=np.complex64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(quartic, complex=True)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

class GoodGradientsCustomGradientFnTest(tf.test.TestCase):

    def test_square_root_has_good_gradients(self) -> None:
        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.complex64)
        with tf.GradientTape(persistent=True) as tape:
            sqrt_p = square_root(p)
        grads = tape.gradient(sqrt_p, p)

        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            sqrt_p = square_root(ensure_complex(p))
        grads = tape.gradient(sqrt_p, p)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)

    def test_cube_root_has_good_gradients(self) -> None:
        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.complex64)
        with tf.GradientTape(persistent=True) as tape:
            cbrt_p = cubic_root(p)
        grads = tape.gradient(cbrt_p, p)

        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            cbrt_p = cubic_root(ensure_complex(p))
        grads = tape.gradient(cbrt_p, p)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsLinearTest(tf.test.TestCase):

    def test_linear_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float32)
        y = tf.Variable([-1, 0, 1], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            a, b, c, d, e = coefs(x, y)
            real_roots = linear(a, b, tape=tape, x=x, y=y)
        grads = tape.gradient(real_roots, y)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsQuadraticTest(tf.test.TestCase):

    def test_quadratic_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float32)
        y = tf.Variable([-1, 0, 1], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            a, b, c, d, e = coefs(x, y)
            roots = quadratic(a, b, c, assume_quadratic=False, tape=tape, x=x, y=y)
            real_roots = tf.stack([tf.math.real(r) for r in roots], axis=-1)
        grads = tape.gradient(real_roots, y)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsCubicTest(tf.test.TestCase):

    def test_cubic_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float32)
        y = tf.Variable([-1, 0, 1], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            a, b, c, d, e = coefs(x, y)
            roots = cubic(a, b, c, d, assume_cubic=False, tape=tape, x=x, y=y)
            real_roots = tf.stack([tf.math.real(r) for r in roots], axis=-1)
        grads = tape.gradient(real_roots, y)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsQuarticTest(tf.test.TestCase):

    def test_quartic_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float32)
        y = tf.Variable([-1, 0, 1], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            a, b, c, d, e = coefs(x, y)
            roots = quartic(a, b, c, d, e, assume_quartic=False, tape=tape, x=x, y=y)
            real_roots = tf.stack([tf.math.real(r) for r in roots], axis=-1)
        grads = tape.gradient(real_roots, y)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    tf.test.main()
