# result-py

A functional programming library for Python that brings **type-safe error handling** through the `Either` monad pattern. Stop using exceptions for control flow—make your errors explicit, composable, and type-checked!

## Features

- **Type-Safe Error Handling**: Errors are values, not exceptions. Your type checker knows exactly what can go wrong.
- **Railway-Oriented Programming**: Chain operations elegantly with `.pipe()` and handle both success and failure paths.
- **Rich Collection Operations**: `map`, `filter`, `reduce`, `flat_map`, and more—all within the Either context.
- **Modern Python**: Built for Python 3.14+ with full type annotations and generics.
- **Functional Composition**: Build complex pipelines from simple, testable functions.

## Installation

```bash
pip install fn-result-py
```

Or with Poetry:

```bash
poetry add fn-result-py
```

## Quick Start

### Basic Usage

```python
from result_py import Either

# Create success and failure values
success = Either.right(42)       # Right = success
failure = Either.left("Error")   # Left = failure

# Chain operations - failures short-circuit automatically
result = (
    Either.right(10)
    .pipe(lambda x: x * 2)       # 20
    .pipe(lambda x: x + 5)       # 25
)
print(result)  # Either(_left=None, _right=25)
```

### Error Handling Made Explicit

```python
from result_py import Either

def divide(a: float, b: float) -> Either[str, float]:
    if b == 0:
        return Either.left("Division by zero!")
    return Either.right(a / b)

def sqrt(x: float) -> Either[str, float]:
    if x < 0:
        return Either.left("Cannot take sqrt of negative number!")
    return Either.right(x ** 0.5)

# Chain operations - first error stops the pipeline
result = (
    Either.right(16.0)
    .pipe(lambda x: divide(x, 2))   # Right(8.0)
    .pipe(sqrt)                      # Right(2.83...)
)

# Handle both cases with match
message = result.match(
    left=lambda err: f"Failed: {err}",
    right=lambda val: f"Result: {val:.2f}"
)
print(message)  # "Result: 2.83"
```

### Working with Collections

```python
from result_py import Either

# Transform collections within Either context
result = (
    Either.right([1, 2, 3, 4, 5])
    .filter(lambda x: x % 2 == 0)     # [2, 4]
    .map(lambda x: x * 10)            # [20, 40]
    .to_list()
)
print(result)  # Either(_left=None, _right=[20, 40])

# Filter and transform in one step
result = (
    Either.right([1, 2, 3, 4, 5])
    .filter_map(lambda x: x * 2 if x > 2 else None)
    .to_list()
)
print(result)  # Either(_left=None, _right=[6, 8, 10])
```

### Wrapping External Code

Use `wrap_external` to safely wrap functions that might throw exceptions:

```python
from result_py import wrap_external
import json

# Wrap a function that might raise exceptions
safe_json_loads = wrap_external(json.loads, json.JSONDecodeError)

result = safe_json_loads('{"valid": "json"}')
print(result)  # Either(_left=None, _right={'valid': 'json'})

result = safe_json_loads('not valid json')
print(result)  # Either(_left=JSONDecodeError(...), _right=None)
```

### Using the `@throws` Decorator

Declare which exceptions your function might throw, and have them automatically converted to `Either.left`:

```python
from result_py import Either, throws

@throws(ValueError, KeyError)
def risky_operation(data: dict, key: str) -> Either[ValueError | KeyError, int]:
    value = data[key]  # Might raise KeyError
    if value < 0:
        raise ValueError("Value must be positive")
    return Either.right(value * 2)

result = risky_operation({"x": 10}, "x")
print(result)  # Either(_left=None, _right=20)

result = risky_operation({}, "x")
print(result)  # Either(_left=KeyError('x'), _right=None)
```

## API Reference

### Creating Either Values

| Method | Description |
|--------|-------------|
| `Either.right(value)` | Create a success value |
| `Either.left(error)` | Create a failure value |
| `Either.success(value)` | Alias for `right` |
| `Either.failure(error)` | Alias for `left` |

### Transformations

| Method | Description |
|--------|-------------|
| `.pipe(f)` | Apply function to right value, supports both `T -> U` and `T -> Either[E, U]` |
| `.and_then(f)` | Monadic bind: chain `T -> Either[E2, U]` functions (alias-like to pipe for Either-returning fns) |
| `.map(f)` | Apply function to each item in an iterable |
| `.map_left(f)` | Transform the left (error) value |
| `.filter(f)` | Filter items in an iterable |
| `.filter_map(f)` | Filter and map in one step (None values filtered out) |
| `.flat_map(f)` | Map and flatten nested iterables |
| `.flatten()` | Flatten nested iterables |

### Tuple Unpacking Variants

| Method | Description |
|--------|-------------|
| `.n_pipe(f)` | Unpack tuple and apply multi-argument function |
| `.n_map(f)` | Map with tuple unpacking (lazy generator) |
| `.n_filter_map(f)` | Filter-map with tuple unpacking (lazy generator) |

### Aggregations

| Method | Description |
|--------|-------------|
| `.reduce(f, initial)` | Reduce iterable to single value |
| `.map_reduce(f, initial)` | Map then reduce |
| `.to_list()` | Convert iterable to list |
| `.to_set()` | Convert iterable to set |
| `.to_counter()` | Count occurrences of items |

### Combining & Matching

| Method | Description |
|--------|-------------|
| `.zip(other)` | Combine two Eithers into tuple |
| `.then(other)` | Chain to next Either if current is Right |
| `.or_else(f)` | Recover from error with `E -> Either[E2, T]` function |
| `.match(left, right)` | Pattern match on Left/Right |
| `.unwrap_or(default)` | Get right value or default |

### Properties

| Property | Description |
|----------|-------------|
| `.is_right` | `True` if this is a Right value |
| `.is_left` | `True` if this is a Left value |

### Utilities

| Method | Description |
|--------|-------------|
| `.partition(f)` | Split iterable into two based on predicate |
| `.to_json()` | Convert right value to JSON string |
| `.write_json_out(path)` | Write Pydantic model to JSON file |
| `.ctx_pipe(f)` | Apply side-effect function, keep original value |

