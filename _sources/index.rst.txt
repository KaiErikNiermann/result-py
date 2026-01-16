.. result-py documentation master file, created by
   sphinx-quickstart on Sat Nov  8 18:39:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

result-py
=========

A functional programming library for Python that brings **type-safe error handling** through the ``Either`` monad pattern.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   api

Why result-py?
--------------

Traditional Python error handling uses exceptions, which are:

- **Invisible**: Nothing in the type signature tells you a function can fail
- **Unstructured**: Any function can throw any exception at any time
- **Hard to compose**: Try/except blocks interrupt your code flow

With ``Either``, errors become **values** that are:

- **Explicit**: The return type tells you exactly what can go wrong
- **Type-checked**: Your IDE and type checker help you handle all cases
- **Composable**: Chain operations naturally with ``.pipe()``

Quick Example
-------------

.. code-block:: python

   from result_py import Either

   def divide(a: float, b: float) -> Either[str, float]:
       if b == 0:
           return Either.left("Division by zero!")
       return Either.right(a / b)

   result = (
       Either.right(100.0)
       .pipe(lambda x: divide(x, 4))   # Right(25.0)
       .pipe(lambda x: divide(x, 5))   # Right(5.0)
       .match(
           left=lambda err: f"Error: {err}",
           right=lambda val: f"Result: {val}"
       )
   )
   print(result)  # "Result: 5.0"

Installation
------------

.. code-block:: bash

   pip install fn-result-py

Or with Poetry:

.. code-block:: bash

   poetry add fn-result-py

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
