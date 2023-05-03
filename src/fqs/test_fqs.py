from typing import Any, Callable, FrozenSet, Set

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
                 complex: bool = False) -> tf.Tensor:
    sampler = lambda: tf.random.uniform((samples, degrees[solver]), lb, ub, dtype=tf.float64)
    real = sampler()
    if complex:
        return tf.dtypes.complex(real, sampler())
    else:
        return real

def coefs_from_roots(roots : tf.Tensor) -> tf.Tensor:
    # returns polynomial coefficients in decreasing order (as typically read left->right)
    # note this is the opposite convention used by np.polynomial, which takes coefs in increasing order
    # ALWAYS RETURN DOUBLE PRECISION FOR LESS VERBOSE TEST CONSTRUCTION DOWNSTREAM
    linears = tf.stack([-roots, tf.ones_like(roots)], axis=-1)
    coefs = linears[0:1, ::-1]
    if roots.dtype.is_complex:
        coefs = tf.bitcast(coefs, roots.dtype.real_dtype)
    else:
        coefs = tf.expand_dims(coefs, -1)
    tail = linears[1:,  :]
    for i in range(tail.shape[0]):
        coefs = tf.pad(coefs, tf.constant([[0, 0], [1, 1], [0, 0]]))
        filters = tf.expand_dims(tail[i,:], -1)
        if filters.dtype.is_complex:
            # signature for filter multiplication in conv1d is
            # specified as r * A where r is a row vector.
            # and complex multiplication of (x + iy) * (w + iz)
            # is equivalent to
            # [w, z] * [[x, y], [-y, x]]
            filters = tf.bitcast(filters, filters.dtype.real_dtype)
            filters = tf.concat([filters, filters[...,::-1]], axis=-2)
            filters = filters * tf.constant([[[ 1, 1],
                                              [-1, 1]]], dtype=filters.dtype)
        else:
            filters = tf.expand_dims(filters, -1)
        coefs = tf.nn.conv1d(coefs, filters, 1, 'VALID')
    # Poly.fromroots(...).coefs is always double precision
    # so for compatibility we will return a double
    coefs = tf.cast(coefs, tf.float64)
    if roots.dtype.is_complex:
        coefs = tf.bitcast(coefs, tf.complex128)
    else:
        coefs = tf.squeeze(coefs, axis=-1)
    return coefs



def _run_one_test(rootses, solver):
    coefs = tf.map_fn(lambda roots: tf.squeeze(coefs_from_roots(roots), 0),
                      rootses,
                      dtype = (tf.complex128 if rootses.dtype.is_complex else tf.float64))
    return tf.concat(solver(*[coefs[:, i:(i + 1)] for i in range(1 + degrees[solver])]), axis=-1)


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
        expected = tf.constant([[i] * 2 for i in range(3)], dtype=tf.float64)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(quadratic)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, tf.cast(expected, tf.complex128), actual)

    def test_identical_complex_roots(self):
        expected = tf.constant([[i - (i * 1j)] * 2 for i in range(3)], dtype=tf.complex128)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(quadratic, complex=True)
        actual = _run_one_test(expected, quadratic)
        assertSetsClose(self, expected, actual)


class CubicTest(tf.test.TestCase):

    def test_identical_real_roots(self):
        expected = tf.constant([[i] * 3 for i in range(3)], dtype=tf.float64)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_real_roots(self):
        expected = tf.constant([[i] + [i + 1] * 2 for i in range(0, 6, 2)], dtype=tf.float64)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(cubic)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, tf.cast(expected, tf.complex128), actual)

    def test_identical_complex_roots(self):
        expected = tf.constant([[i + ((i + 1) * 1j)] * 3 for i in range(3)], dtype=tf.complex128)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_two_identical_complex_roots(self):
        expected = tf.constant([[-i + ((i + 1) * 1j)] + [i - ((i + 1) * 1j)] * 2 for i in range(3)], dtype=tf.complex128)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(cubic, complex=True)
        actual = _run_one_test(expected, cubic)
        assertSetsClose(self, expected, actual)


class QuarticTest(tf.test.TestCase):

    def test_identical_real_roots(self):
        expected = tf.constant([[i] * 4 for i in range(3)], dtype=tf.float64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_three_identical_real_roots(self):
        expected = tf.constant([[i] + [i + 1] * 3 for i in range(0, 6, 2)], dtype=tf.float64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_biquadratic_real_roots(self):
        expected = tf.constant([[i] * 2 + [i + 1] * 2 for i in range(0, 6, 2)], dtype=tf.float64)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_distinct_real_roots(self):
        expected = random_roots(quartic)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, tf.cast(expected, tf.complex128), actual)

    def test_identical_complex_roots(self):
        expected = tf.constant([[i + ((i + 1) * 1j)] * 4 for i in range(3)], dtype=tf.complex128)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_one_distinct_three_identical_complex_roots(self):
        expected = tf.constant([[-i + ((i + 1) * 1j)] + [i - ((i + 1) * 1j)] * 3 for i in range(3)], dtype=tf.complex128)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_biquadratic_complex_roots(self):
        expected = tf.constant([[-i + ((i + 1) * 1j)] * 2 + [i - ((i + 1) * 1j)] * 2 for i in range(3)],
                            dtype=tf.complex128)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

    def test_distinct_complex_roots(self):
        expected = random_roots(quartic, complex=True)
        actual = _run_one_test(expected, quartic)
        assertSetsClose(self, expected, actual)

class GoodGradientsCustomGradientFnTest(tf.test.TestCase):

    def test_square_root_has_good_gradients(self) -> None:
        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.complex128)
        with tf.GradientTape(persistent=True) as tape:
            sqrt_p = square_root(p)
        grads = tape.gradient(sqrt_p, p)

        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            sqrt_p = square_root(ensure_complex(p))
        grads = tape.gradient(sqrt_p, p)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)

    def test_cube_root_has_good_gradients(self) -> None:
        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.complex128)
        with tf.GradientTape(persistent=True) as tape:
            cbrt_p = cubic_root(p)
        grads = tape.gradient(cbrt_p, p)

        p = tf.Variable([-1, -1e-10, 0, 1e-4, 1], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            cbrt_p = cubic_root(ensure_complex(p))
        grads = tape.gradient(cbrt_p, p)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsLinearTest(tf.test.TestCase):

    def test_linear_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float64)
        y = tf.Variable([-1, 0, 1], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            a, b, c, d, e = coefs(x, y)
            real_roots = linear(a, b, tape=tape, x=x, y=y)
        grads = tape.gradient(real_roots, y)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsQuadraticTest(tf.test.TestCase):

    def test_quadratic_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float64)
        y = tf.Variable([-1, 0, 1], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            a, b, c, d, e = coefs(x, y)
            roots = quadratic(a, b, c, assume_quadratic=False, tape=tape, x=x, y=y)
            real_roots = tf.stack([tf.math.real(r) for r in roots], axis=-1)
        grads = tape.gradient(real_roots, y)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsCubicTest(tf.test.TestCase):

    def test_cubic_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float64)
        y = tf.Variable([-1, 0, 1], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            a, b, c, d, e = coefs(x, y)
            roots = cubic(a, b, c, d, assume_cubic=False, tape=tape, x=x, y=y)
            real_roots = tf.stack([tf.math.real(r) for r in roots], axis=-1)
        grads = tape.gradient(real_roots, y)
        tf.debugging.assert_all_finite(grads, 'grad real')
        self.assertNotAllClose(grads, 0.)


class GoodGradientsQuarticTest(tf.test.TestCase):

    def test_quartic_solver_has_good_gradients(self):
        x = tf.Variable([0, 0, 0], dtype=tf.float64)
        y = tf.Variable([-1, 0, 1], dtype=tf.float64)
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
