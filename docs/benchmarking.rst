:github_url: https://github.com/marcdemers/py_vollib_vectorized

Performance
============

With :obj:`numba`, `py_vollib_vectorized` provides a speed boost to the Black/Black-Scholes/Black-Scholes-Merton models when compared to traditional for-loops, and even to other iterative and vectorized implementation.
The calculation scales well with the number of option contracts.
You can price millions of option contracts in a matter of milliseconds.

The figure below shows the time to calculate the option prices and implied volatilities for a fixed number of contracts.
We capped the runtime at 60 seconds.
While this performance grah was done with option prices and IVs, all functions in this library benefit from this speed boost.
As such, a similar comparison would be obtained with other py_vollib_vectorized functions.

.. include:: _static/benchmark_table.rst

.. image:: _static/benchmark.png
   :width: 600
