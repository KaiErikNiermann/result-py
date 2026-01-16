API Reference
=============

Either Class
------------

The ``Either`` type represents a value that can be one of two possibilities:

- **Left**: Conventionally used for errors/failures
- **Right**: Conventionally used for success values

Creating Either Values
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from result_py import Either

   # Success values
   success = Either.right(42)
   success = Either.success(42)  # Alias

   # Failure values  
   failure = Either.left("Error message")
   failure = Either.failure("Error message")  # Alias

.. automodule:: result_py.either
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. autoclass:: result_py.either.Either
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

Helper Functions
----------------

as_either
~~~~~~~~~

Wraps a value in ``Either.right``:

.. code-block:: python

   from result_py import as_either

   result = as_either(42)  # Either(_left=None, _right=42)

wrap_external
~~~~~~~~~~~~~

Wraps a function that might throw exceptions:

.. code-block:: python

   from result_py import wrap_external
   import json

   safe_loads = wrap_external(json.loads, json.JSONDecodeError)
   
   result = safe_loads('{"key": "value"}')  # Right({'key': 'value'})
   result = safe_loads('invalid')           # Left(JSONDecodeError(...))

throws Decorator
~~~~~~~~~~~~~~~~

Decorator to catch specified exceptions and convert them to ``Either.left``:

.. code-block:: python

   from result_py import Either, throws

   @throws(ValueError, KeyError)
   def process(data: dict) -> Either[ValueError | KeyError, int]:
       return Either.right(data["key"] * 2)

   result = process({"key": 5})  # Right(10)
   result = process({})          # Left(KeyError('key'))

Contracts Module
----------------

.. automodule:: result_py.contracts
   :members:
   :undoc-members:
   :show-inheritance:
