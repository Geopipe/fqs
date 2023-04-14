# FQS: Fast Quartic and Cubic solver

A fast python function for computing roots of a quartic equation (4th order polynomial) and a cubic equation (3rd order polynomial) in tensorflow.
Works properly even when the polynomial coefficients are complex.

The original numpy version is at [NKrvavica/fqs](https://github.com/NKrvavica/fqs)


# Features

 * The function is optimized for computing single or multiple roots of 3rd and 4th order polynomials (cubic and quartic equations).
 * A closed-form analytical solutions are used for roots of cubic and quartic equations.
 * Implemented in tensorflow, supports solving many polynomials in parallel. 
 
 # Requirements
 
 Python 3+, tensorflow
 
 
 # Usage

All functions are found in `fqs.py`, which can be cloned to local folder.
 
See [test_fqs.py](test_fqs.py) for unit tests and example usage.

 
 # FAQ

 > Why not simply use `numpy.roots` or `numpy.linalg.eigvals` for all polynomials?
 
For single polynomial, both quartic and cubic solvers are one order of magnitude faster than `numpy.roots` and `numpy.linalg.eigvals`. 
For large number of polynomials (>1_0000), both quartic and cubic solver are one order of magnitude faster than `numpy.linalg.eigvals` and two order of magnitude faster than `numpy.roots` inside a list comprehension.

> Why did you create this fork?

To support root finding within a tensorflow graph.
 
 
 # License
 
[MIT license](LICENSE)


