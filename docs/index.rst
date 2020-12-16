:github_url: https://github.com/marcdemers/py_vollib_vectorized

.. role:: underline
    :class: underline

py_vollib_vectorized documentation
================================================

The `py_vollib_vectorized` library provides an easy and intuitive interface for pricing thousands of option contracts and calculating greeks.
It is built on top of the `py_vollib` library, and provides an API to the patches.
Upon import, it will automatically patch the corresponding `py_vollib` functions to support vectorization.
Inputs can then be passed as tuples, lists, `numpy.array`s, `pandas.Series` or `pandas.DataFrame`s.

On top of vectorization, this library includes a number of additional speedups (see Performance).

.. toctree::
    :glob:
    :maxdepth: 1

    installation
    quick_start


.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Package Reference:

    pkg_ref/models
    pkg_ref/iv
    pkg_ref/greeks


.. toctree::
    :glob:
    :maxdepth: 1
    :caption: API:

    api/api


.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Patching py_vollib:


.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Benchmarking:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

