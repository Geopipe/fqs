from typing import Tuple

import tensorflow as tf


def coefs(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generates polynomial coefficients for performing an orthogonal (or total) least-quartics regression
    for a line given by $y = x \tan(t)$

    The signed point-to-line distance is defined as $x_i*\sin(t) - y_i*\cos(t)$ for a single point
    so we must minimize $f(\vec{x}, \vec{y}, t) = \sum^n_{i=1} (x_i \sin(t) - y_i \cos(t))^4$
    with respective to $t$.

    The extrema of this polynomial are at the roots and ±∞, and solve for the roots as
    $\frac{\partial f(\vec{x}, \vec{y}, t)}{\partial t} = \sum^n_{i=1} 4 (x_i \sin(t) - y_i \cos(t))^3 (x_i \cos(t) + y_i \sin(t) = 0$


    Solving for $t$, we get
    $\tan(t) \in \{\pm \infty\} \cup \mathrm{RootsOf}_Z \sum^n_{i=1} \left ( x_i^3 y_i Z^4 + (x_i^4 - 3 x_i^2 y_i^2) Z^3 + 3 (x_i y_i^3 - x_i^3 y_i) Z^2  + (3 x_i^2 y_i^2 - y_i^4) Z - x_i y_i^3 \right )$

    Thus this function returns the coefficients of the polynomial in $Z$.
    """
    x2 = x * x
    y2 = y * y
    xy = x * y

    sigma = lambda z: tf.reduce_sum(z, axis=-1)

    x4 = sigma(x2 * x2)
    x3y = sigma(x2 * xy)
    x2y2 = sigma(xy * xy)
    xy3 = sigma(xy * y2)
    y4 = sigma(y2 * y2)

    a = x3y
    b = x4 - 3 * x2y2
    c = 3 * (xy3 - x3y)
    d = 3 * x2y2 - y4
    e = -xy3
    return a, b, c, d, e
