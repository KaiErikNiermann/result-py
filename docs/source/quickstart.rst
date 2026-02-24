Quickstart
==========

Creating Either Values
----------------------

``Either`` has two sides: **Right** for success, **Left** for failure.

.. code-block:: python

   from result_py import Either

   success = Either.right(42)
   failure = Either.left("something went wrong")

   # Aliases
   success = Either.success(42)
   failure = Either.failure("something went wrong")

Chaining with pipe
------------------

``.pipe()`` applies a function to the Right value. If the Either is Left, the
chain short-circuits and the error propagates untouched.

.. code-block:: python

   result = (
       Either.right(5)
       .pipe(lambda x: x + 1)   # 6
       .pipe(lambda x: x * 2)   # 12
       .pipe(lambda x: x - 3)   # 9
   )
   # Either(_left=None, _right=9)

If any step returns an ``Either.left``, everything after it is skipped:

.. code-block:: python

   def safe_div(x: float) -> Either[str, float]:
       if x == 0:
           return Either.left("cannot divide by zero")
       return Either.right(100 / x)

   result = (
       Either.right(0)
       .pipe(safe_div)           # Left("cannot divide by zero")
       .pipe(lambda x: x + 1)   # skipped
   )

Extracting Values
-----------------

``match`` lets you handle both cases:

.. code-block:: python

   message = result.match(
       left=lambda err: f"Failed: {err}",
       right=lambda val: f"Got: {val}",
   )

``unwrap_or`` gives you the Right value or a default:

.. code-block:: python

   value = Either.right(42).unwrap_or(0)   # 42
   value = Either.left("err").unwrap_or(0) # 0

Working with Collections
------------------------

When an Either wraps an iterable, you get collection operations:

.. code-block:: python

   result = (
       Either.right([1, 2, 3, 4, 5])
       .filter(lambda x: x % 2 == 0)      # [2, 4]
       .map(lambda x: x * 10)             # [20, 40]
       .to_list()
   )
   # Either(_left=None, _right=[20, 40])
