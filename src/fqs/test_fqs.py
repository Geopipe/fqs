from typing import Any, Callable, FrozenSet, Set

from .fqs import cubic
from .fqs import cubic_root
from .fqs import ensure_complex
from .fqs import linear
from .fqs import quadratic
from .fqs import quartic
from .fqs import square_root
import tensorflow as tf

degrees = {linear: 1, quadratic: 2, cubic: 3, quartic: 4}


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

class NthRootsCustomGradientsFnTest(tf.test.TestCase):

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


class DifferentiableSolversTest(tf.test.TestCase):
    """
    For an Nth degree polynomial, define N roots as variables,
    create polynomial coefficients, solve, and then compute
    the matrix of partial derivatives of the solved roots w.r.t
    the input roots variables.
    Summing these partial derivatives over input variables should
    yield 1 for each solved root.

    We should also see various interpretable intermediate values
    for repeated roots, although these get less interpretable as
    the degree of the polynomial increases.
    """
    def test_linear_solver_has_good_gradients(self):
        x = tf.Variable([-2,-1, 0, 0, 1, 3], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            roots = tf.expand_dims(x, axis=-1)
            derived_roots = _run_one_test(roots, linear)
            derived_x = derived_roots[:, 0]
            dr_dx = tape.gradient(derived_x, x)
        self.assertAllClose(tf.ones_like(dr_dx), dr_dx)

    def test_quadratic_solver_has_good_gradients(self):
        x = tf.Variable([-2,-1, 0, 0, 1, 3], dtype=tf.float64)
        y = tf.Variable([ 2, 1, 0, 1, 4, 3], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            roots = tf.stack([x, y],axis=-1)
            derived_roots = _run_one_test(roots, quadratic)
            derived_x = derived_roots[:, 0]
            derived_y = derived_roots[:, 1]
        dr_dx = tf.stack([tape.gradient(derived_x, x), tape.gradient(derived_y, x)], axis=0)
        dr_dy = tf.stack([tape.gradient(derived_x, y), tape.gradient(derived_y, y)], axis=0)
        all_grads = tf.stack([dr_dx, dr_dy], axis=0)

        same_mask = tf.equal(x, y)
        expected_same = tf.cast(tf.where(same_mask, 0.5, 1.0), dtype=tf.float64)
        expected_diff = tf.cast(tf.where(same_mask, 0.5, 0.0), dtype=tf.float64)
        all_expected = tf.stack([expected_same, expected_diff], axis=0)
        all_expected = tf.stack([all_expected, all_expected[::-1,...]], axis=0)
        self.assertAllClose(all_expected, all_grads, atol=1e-04)

    def test_cubic_solver_has_good_gradients(self):
        x = tf.Variable([-2,-1, 0,-1, 1, 3], dtype=tf.float64)
        y = tf.Variable([ 2, 0, 0,-1, 4, 5], dtype=tf.float64)
        z = tf.Variable([ 2, 1, 0, 0, 5, 6], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            roots = tf.stack([x, y, z],axis=-1)
            derived_roots = _run_one_test(roots, cubic)
            derived_roots = tf.sort(tf.math.real(derived_roots), axis=-1)
            derived_x = derived_roots[:, 0]
            derived_y = derived_roots[:, 1]
            derived_z = derived_roots[:, 2]
        dr_dx = tf.stack([tape.gradient(derived_x, x), tape.gradient(derived_y, x), tape.gradient(derived_z, x)], axis=0)
        dr_dy = tf.stack([tape.gradient(derived_x, y), tape.gradient(derived_y, y), tape.gradient(derived_z, y)], axis=0)
        dr_dz = tf.stack([tape.gradient(derived_x, z), tape.gradient(derived_y, z), tape.gradient(derived_z, z)], axis=0)
        all_grads = tf.stack([dr_dx, dr_dy, dr_dz], axis=0)

        actual = tf.reduce_sum(all_grads, axis=0)

        self.assertAllClose(tf.ones_like(actual), actual, atol=1e-04)

    def test_quartic_solver_has_good_gradients(self):
        x = tf.Variable([-2,-1, 0,-1, 1, 3], dtype=tf.float64)
        y = tf.Variable([-2, 0, 0,-1, 4, 5], dtype=tf.float64)
        z = tf.Variable([ 2, 0, 0,-1, 8, 7], dtype=tf.float64)
        w = tf.Variable([ 2, 1, 0, 0,16,11], dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:
            roots = tf.stack([x, y, z, w],axis=-1)
            derived_roots = _run_one_test(roots, quartic)
            derived_roots = tf.sort(tf.math.real(derived_roots), axis=-1)
            derived_x = derived_roots[:, 0]
            derived_y = derived_roots[:, 1]
            derived_z = derived_roots[:, 2]
            derived_w = derived_roots[:, 3]
        dr_dx = tf.stack([tape.gradient(derived_x, x), tape.gradient(derived_y, x), tape.gradient(derived_z, x), tape.gradient(derived_w, x)], axis=0)
        dr_dy = tf.stack([tape.gradient(derived_x, y), tape.gradient(derived_y, y), tape.gradient(derived_z, y), tape.gradient(derived_w, y)], axis=0)
        dr_dz = tf.stack([tape.gradient(derived_x, z), tape.gradient(derived_y, z), tape.gradient(derived_z, z), tape.gradient(derived_w, z)], axis=0)
        dr_dw = tf.stack([tape.gradient(derived_x, w), tape.gradient(derived_y, w), tape.gradient(derived_z, w), tape.gradient(derived_w, w)], axis=0)
        all_grads = tf.stack([dr_dx, dr_dy, dr_dz, dr_dw], axis=0)

        actual = tf.reduce_sum(all_grads, axis=0)

        self.assertAllClose(tf.ones_like(actual), actual, atol=1e-04)

if __name__ == "__main__":
    tf.config.run_functions_eagerly(False)
    tf.test.main()
