User Guide
==========

This page covers the main patterns you will use with result-py.

Piping and Chaining
-------------------

``.pipe()`` is the primary way to transform values inside an Either. It accepts
both plain functions (``T -> U``) and Either-returning functions
(``T -> Either[E, U]``). In the second case, the returned Either is
"flattened" so you don't end up with nested Eithers.

.. code-block:: python

   from result_py import Either

   def parse_int(s: str) -> Either[str, int]:
       try:
           return Either.right(int(s))
       except ValueError:
           return Either.left(f"'{s}' is not an integer")

   result = (
       Either.right("42")
       .pipe(parse_int)              # Right(42)
       .pipe(lambda x: x * 2)       # Right(84)
   )

``.and_then()`` is an explicit monadic bind — it only accepts
``T -> Either[E, U]`` functions, which can be useful for clarity when every
step is fallible.

Tuple Unpacking (n_pipe, n_map)
-------------------------------

When your Either holds a tuple, the ``n_``-prefixed methods unpack it:

.. code-block:: python

   result = Either.right((10, 20)).n_pipe(lambda x, y: x + y)
   # Either(_left=None, _right=30)

   result = (
       Either.right([(1, 2), (3, 4)])
       .n_map(lambda x, y: x + y)
       .to_list()
   )
   # Either(_left=None, _right=[3, 7])

Collection Operations
---------------------

These methods operate on ``Either[L, Iterable[T]]``.

**map / filter / filter_map**

.. code-block:: python

   (Either.right([1, 2, 3, 4, 5])
       .filter(lambda x: x > 2)                          # [3, 4, 5]
       .map(lambda x: x ** 2)                             # [9, 16, 25]
       .to_list())

   (Either.right([1, 2, 3, 4, 5])
       .filter_map(lambda x: x * 10 if x % 2 == 0 else None)  # [20, 40]
       .to_list())

**flat_map / flatten**

.. code-block:: python

   (Either.right([1, 2, 3])
       .flat_map(lambda x: [x, x * 10])   # [1, 10, 2, 20, 3, 30]
       .to_list())

   (Either.right([[1, 2], [3, 4]])
       .flatten()                          # [1, 2, 3, 4]
       .to_list())

**reduce / map_reduce**

.. code-block:: python

   # Sum with reduce
   Either.right([1, 2, 3, 4]).reduce(lambda acc, x: acc + x, 0)
   # Either(_left=None, _right=10)

   # Map then reduce in one step
   Either.right([1, 2, 3]).map_reduce(lambda x: x * 2, 0)
   # Either(_left=None, _right=12)

**partition**

Split an iterable into two groups based on a predicate:

.. code-block:: python

   truthy, falsy = (
       Either.right([1, 2, 3, 4, 5])
       .partition(lambda x: x > 3)
       .unwrap_or(([], []))
   )
   # truthy = [4, 5], falsy = [1, 2, 3]

Conversions
^^^^^^^^^^^

- ``.to_list()`` — materialise the iterable into a list
- ``.to_set()`` — into a set
- ``.to_counter()`` / ``.counted()`` — into a ``Counter``
- ``.to_items()`` — extract ``.items()`` from a Mapping
- ``.to_json()`` — serialise the Right value to a JSON string

Error Handling
--------------

**map_left** — transform the error value:

.. code-block:: python

   Either.left("bad input").map_left(lambda e: f"Error: {e}")
   # Either(_left="Error: bad input", _right=None)

**or_else** — attempt recovery from an error:

.. code-block:: python

   def recover(err: str) -> Either[int, str]:
       if err == "retry":
           return Either.right("recovered")
       return Either.left(500)

   Either.left("retry").or_else(recover)
   # Either(_left=None, _right="recovered")

Wrapping External Code
----------------------

**wrap_external** turns a regular function into one that returns Either
instead of raising:

.. code-block:: python

   import json
   from result_py import wrap_external

   safe_loads = wrap_external(json.loads, json.JSONDecodeError)

   safe_loads('{"a": 1}')   # Right({'a': 1})
   safe_loads('invalid')    # Left(JSONDecodeError(...))

**@throws** is a decorator that does the same thing inline:

.. code-block:: python

   from result_py import Either, throws

   @throws(ValueError, KeyError)
   def process(data: dict) -> Either[ValueError | KeyError, int]:
       return Either.right(data["key"] * 2)

   process({"key": 5})  # Right(10)
   process({})          # Left(KeyError('key'))

Both will also catch unexpected exceptions and emit a warning rather than
silently swallowing them.

Combining Eithers
-----------------

**zip** — combine two Eithers into a tuple:

.. code-block:: python

   a = Either.right(1)
   b = Either.right("hello")
   a.zip(b)  # Either(_left=None, _right=(1, 'hello'))

**then** — sequence: keep the second if the first is Right:

.. code-block:: python

   Either.right("ok").then(Either.right(42))
   # Either(_left=None, _right=42)

Side Effects
------------

**ctx_pipe** runs a function for its side effect, then passes the original
value through unchanged:

.. code-block:: python

   (Either.right(42)
       .ctx_pipe(lambda x: print(f"debug: {x}"))  # prints "debug: 42"
       .pipe(lambda x: x + 1))
   # Either(_left=None, _right=43)

Progress Tracking
-----------------

``map``, ``filter``, ``reduce``, ``map_reduce``, ``n_map``, and
``filter_map`` all accept ``track=True`` to wrap the iterable in a
``tqdm`` progress bar:

.. code-block:: python

   (Either.right(range(1000))
       .map(expensive_fn, track=True, desc="Processing")
       .to_list())
