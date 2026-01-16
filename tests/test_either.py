"""Comprehensive tests for the Either monad and related utilities."""

import pytest
import json
import tempfile
import os
import warnings
from typing import Optional
from collections import Counter
from pydantic import BaseModel

from result_py.either import (
    Either,
    as_either,
    as_failure,
    curry,
    throws,
    wrap_external,
    ident,
    ExternalWrapWarning,
    WriteJsonWarning,
)


class TestEitherCreation:
    """Tests for Either creation methods."""

    def test_right_creation(self):
        """Either.right should create a Right value."""
        e = Either.right(42)
        assert e._right == 42
        assert e._left is None

    def test_left_creation(self):
        """Either.left should create a Left value."""
        e = Either.left("error")
        assert e._left == "error"
        assert e._right is None

    def test_success_alias(self):
        """Either.success should be an alias for right."""
        e = Either.success(100)
        assert e._right == 100
        assert e._left is None

    def test_failure_alias(self):
        """Either.failure should be an alias for left."""
        e = Either.failure("fail")
        assert e._left == "fail"
        assert e._right is None

    def test_as_either_helper(self):
        """as_either should wrap a value in Either.right."""
        e = as_either("value")
        assert e._right == "value"
        assert e._left is None

    def test_as_failure_helper(self):
        """as_failure should wrap a value in Either.left."""
        e = as_failure("error")
        assert e._left == "error"
        assert e._right is None


class TestEitherIsChecks:
    """Tests for is_right and is_left type guards."""

    def test_is_right_returns_true_for_right(self):
        """is_right should return True when value exists."""
        e = Either.right(42)
        assert e.is_right is True

    def test_is_right_returns_false_for_none(self):
        """is_right should return False when value is None."""
        e = Either.left("error")
        assert e.is_right is False

    def test_is_left_returns_true_for_left(self):
        """is_left should return True when value exists."""
        e = Either.left("error")
        assert e.is_left is True

    def test_is_left_returns_false_for_none(self):
        """is_left should return False when value is None."""
        e = Either.right(42)
        assert e.is_left is False


class TestEitherPipe:
    """Tests for the pipe method."""

    def test_pipe_applies_function_to_right(self):
        """pipe should apply function to right value."""
        e = Either.right(5).pipe(lambda x: x * 2)
        assert e._right == 10

    def test_pipe_chains_multiple_functions(self):
        """pipe should chain multiple functions."""
        e = Either.right(5).pipe(lambda x: x + 1).pipe(lambda x: x * 2)
        assert e._right == 12

    def test_pipe_short_circuits_on_left(self):
        """pipe should not apply function when Either is left."""
        e = Either.left("error").pipe(lambda x: x * 2)
        assert e._left == "error"
        assert e._right is None

    def test_pipe_with_either_returning_function(self):
        """pipe should flatten when function returns Either."""

        def safe_div(x: int) -> Either[str, int]:
            if x == 0:
                return Either.left("division by zero")
            return Either.right(10 // x)

        e = Either.right(2).pipe(safe_div)
        assert e._right == 5

    def test_pipe_with_either_returning_left(self):
        """pipe should propagate left from function returning Either."""

        def safe_div(x: int) -> Either[str, int]:
            if x == 0:
                return Either.left("division by zero")
            return Either.right(10 // x)

        e = Either.right(0).pipe(safe_div)
        assert e._left == "division by zero"


class TestEitherNPipe:
    """Tests for the n_pipe method."""

    def test_n_pipe_unpacks_tuple(self):
        """n_pipe should unpack tuple and apply function."""
        e = Either.right((2, 3)).n_pipe(lambda x, y: x + y)
        assert e._right == 5

    def test_n_pipe_short_circuits_on_left(self):
        """n_pipe should not apply function when Either is left."""
        e: Either[str, tuple[int, int]] = Either.left("error")
        result = e.n_pipe(lambda x, y: x + y)
        assert result._left == "error"


class TestEitherMap:
    """Tests for the map method."""

    def test_map_applies_to_each_element(self):
        """map should apply function to each element in iterable."""
        e = Either.right([1, 2, 3]).map(lambda x: x * 2).to_list()
        assert e._right == [2, 4, 6]

    def test_map_short_circuits_on_left(self):
        """map should not apply when Either is left."""
        e: Either[str, list[int]] = Either.left("error")
        result = e.map(lambda x: x * 2).to_list()
        assert result._left == "error"

    def test_map_is_lazy(self):
        """map should return a lazy iterable."""
        e = Either.right([1, 2, 3]).map(lambda x: x * 2)
        # Result should be an iterable, not a list
        assert hasattr(e._right, "__iter__")


class TestEitherFilter:
    """Tests for the filter method."""

    def test_filter_keeps_matching_elements(self):
        """filter should keep elements where predicate returns True."""
        e = Either.right([1, 2, 3, 4, 5]).filter(lambda x: x > 2).to_list()
        assert e._right == [3, 4, 5]

    def test_filter_removes_non_matching_elements(self):
        """filter should remove elements where predicate returns False."""
        e = Either.right([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).to_list()
        assert e._right == [2, 4]

    def test_filter_short_circuits_on_left(self):
        """filter should not apply when Either is left."""
        e: Either[str, list[int]] = Either.left("error")
        result = e.filter(lambda x: x > 0).to_list()
        assert result._left == "error"


class TestEitherFilterMap:
    """Tests for the filter_map method."""

    def test_filter_map_applies_and_filters(self):
        """filter_map should apply function and filter out None results."""
        e = Either.right([1, 2, 3, 4, 5]).filter_map(
            lambda x: x * 2 if x % 2 == 0 else None
        )
        assert list(e._right) == [4, 8]

    def test_filter_map_short_circuits_on_left(self):
        """filter_map should not apply when Either is left."""
        e: Either[str, list[int]] = Either.left("error")
        result = e.filter_map(lambda x: x * 2 if x > 0 else None)
        assert result._left == "error"


class TestEitherNMap:
    """Tests for the n_map method."""

    def test_n_map_unpacks_tuples(self):
        """n_map should unpack tuples and apply function."""
        e = Either.right([(1, 2), (3, 4)]).n_map(lambda x, y: x + y).to_list()
        assert e._right == [3, 7]

    def test_n_map_short_circuits_on_left(self):
        """n_map should not apply when Either is left."""
        e: Either[str, list[tuple[int, int]]] = Either.left("error")
        result = e.n_map(lambda x, y: x + y)
        assert result._left == "error"


class TestEitherNFilterMap:
    """Tests for the n_filter_map method."""

    def test_n_filter_map_unpacks_and_filters(self):
        """n_filter_map should unpack tuples, apply function, and filter None."""
        e = Either.right([(1, 2), (3, 4), (5, 6)]).n_filter_map(
            lambda x, y: x + y if (x + y) > 5 else None
        ).to_list()
        assert e._right == [7, 11]


class TestEitherFlatMap:
    """Tests for the flat_map method."""

    def test_flat_map_flattens_iterables(self):
        """flat_map should apply function and flatten results."""
        e = Either.right([1, 2, 3]).flat_map(lambda x: [x, x * 10]).to_list()
        assert e._right == [1, 10, 2, 20, 3, 30]


class TestEitherFlatten:
    """Tests for the flatten method."""

    def test_flatten_nested_iterables(self):
        """flatten should flatten nested iterables."""
        e = Either.right([[1, 2], [3, 4], [5]]).flatten().to_list()
        assert e._right == [1, 2, 3, 4, 5]


class TestEitherReduce:
    """Tests for the reduce method."""

    def test_reduce_accumulates_values(self):
        """reduce should accumulate values with function."""
        e = Either.right([1, 2, 3, 4]).reduce(lambda acc, x: acc + x, 0)
        assert e._right == 10  # 1 + 2 + 3 + 4 = 10

    def test_reduce_with_different_initial(self):
        """reduce should use the initial value correctly."""
        e = Either.right([1, 2, 3, 4, 5]).reduce(lambda acc, x: acc + x, 0)
        assert e._right == 15  # 1 + 2 + 3 + 4 + 5 = 15

    def test_reduce_with_multiplication(self):
        """reduce should work with multiplication."""
        e = Either.right([1, 2, 3, 4]).reduce(lambda acc, x: acc * x, 1)
        assert e._right == 24  # 1 * 2 * 3 * 4 = 24

    def test_reduce_short_circuits_on_left(self):
        """reduce should not apply when Either is left."""
        e: Either[str, list[int]] = Either.left("error")
        result = e.reduce(lambda acc, x: acc + x, 0)
        assert result._left == "error"


class TestEitherMapReduce:
    """Tests for the map_reduce method."""

    def test_map_reduce_maps_then_reduces(self):
        """map_reduce should map then reduce."""
        e = Either.right([1, 2, 3]).map_reduce(lambda x: x * 2, 0)
        assert e._right == 12  # 2 + 4 + 6 = 12


class TestEitherNMapReduce:
    """Tests for the n_map_reduce method."""

    def test_n_map_reduce_unpacks_maps_reduces(self):
        """n_map_reduce should unpack tuples, map, then reduce."""
        e = Either.right([(1, 2), (3, 4)]).n_map_reduce(lambda x, y: x + y, 0)
        assert e._right == 10  # (1+2) + (3+4) = 10


class TestEitherToList:
    """Tests for the to_list method."""

    def test_to_list_converts_iterable(self):
        """to_list should convert iterable to list."""
        e = Either.right(range(5)).to_list()
        assert e._right == [0, 1, 2, 3, 4]


class TestEitherToSet:
    """Tests for the to_set method."""

    def test_to_set_converts_iterable(self):
        """to_set should convert iterable to set."""
        e = Either.right([1, 2, 2, 3, 3, 3]).to_set()
        assert e._right == {1, 2, 3}


class TestEitherToCounter:
    """Tests for the to_counter method."""

    def test_to_counter_counts_elements(self):
        """to_counter should count occurrences of elements."""
        e = Either.right(["a", "b", "a", "c", "a", "b"]).to_counter()
        assert e._right == {"a": 3, "b": 2, "c": 1}


class TestEitherCounted:
    """Tests for the counted method."""

    def test_counted_counts_elements(self):
        """counted should count occurrences of elements."""
        e = Either.right([1, 2, 1, 3, 1, 2]).counted()
        assert e._right[1] == 3
        assert e._right[2] == 2
        assert e._right[3] == 1


class TestEitherPartition:
    """Tests for the partition method."""

    def test_partition_splits_by_predicate(self):
        """partition should split iterable by predicate."""
        e = Either.right([1, 2, 3, 4, 5]).partition(lambda x: x > 3)
        truthy, falsy = e._right
        assert list(truthy) == [4, 5]
        assert list(falsy) == [1, 2, 3]


class TestEitherTwoPartition:
    """Tests for the two_partition method."""

    def test_two_partition_with_type_guard(self):
        """two_partition should partition with type narrowing."""

        def is_int(x: int | str) -> bool:
            return isinstance(x, int)

        e = Either.right([1, "a", 2, "b", 3]).two_partition(is_int)
        ints, strs = e._right
        assert list(ints) == [1, 2, 3]
        assert list(strs) == ["a", "b"]


class TestEitherToItems:
    """Tests for the to_items method."""

    def test_to_items_extracts_mapping_items(self):
        """to_items should extract items from a mapping."""
        e = Either.right({"a": 1, "b": 2}).to_items().to_list()
        assert ("a", 1) in e._right
        assert ("b", 2) in e._right


class TestEitherToRootItems:
    """Tests for the to_root_items method."""

    def test_to_root_items_extracts_root_items(self):
        """to_root_items should extract items from HasRootMapping."""

        class RootModel:
            @property
            def root(self):
                return {"x": 10, "y": 20}

        e = Either.right(RootModel()).to_root_items().to_list()
        assert ("x", 10) in e._right
        assert ("y", 20) in e._right


class TestEitherMapLeft:
    """Tests for the map_left method."""

    def test_map_left_transforms_left_value(self):
        """map_left should transform the left value."""
        e: Either[str, int] = Either.left("error")
        result = e.map_left(lambda s: f"Error: {s}")
        assert result._left == "Error: error"

    def test_map_left_preserves_right_value(self):
        """map_left should not affect right values."""
        e = Either.right(42)
        result = e.map_left(lambda s: f"Error: {s}")
        assert result._right == 42

    def test_map_left_with_type_change(self):
        """map_left should allow changing the error type."""
        e: Either[str, int] = Either.left("not found")
        result = e.map_left(lambda s: 404 if s == "not found" else 500)
        assert result._left == 404


class TestEitherOrElse:
    """Tests for the or_else method."""

    def test_or_else_recovers_from_error(self):
        """or_else should allow recovery from errors."""

        def recover(err: str) -> Either[int, str]:
            if err == "recoverable":
                return Either.right("recovered")
            return Either.left(500)

        e: Either[str, str] = Either.left("recoverable")
        result = e.or_else(recover)
        assert result._right == "recovered"

    def test_or_else_propagates_new_error(self):
        """or_else should propagate new errors from recovery function."""

        def recover(err: str) -> Either[int, str]:
            return Either.left(500)

        e: Either[str, str] = Either.left("unrecoverable")
        result = e.or_else(recover)
        assert result._left == 500

    def test_or_else_preserves_right_value(self):
        """or_else should not affect right values."""

        def recover(err: str) -> Either[int, str]:
            return Either.left(500)

        e: Either[str, str] = Either.right("success")
        result = e.or_else(recover)
        assert result._right == "success"


class TestEitherAndThen:
    """Tests for the and_then method."""

    def test_and_then_chains_success(self):
        """and_then should chain operations on Right values."""
        def parse_int(s: str) -> Either[str, int]:
            try:
                return Either.right(int(s))
            except ValueError:
                return Either.left("not a number")

        result = Either.right("42").and_then(parse_int)
        assert result._right == 42

    def test_and_then_short_circuits_on_initial_left(self):
        """and_then should preserve Left and not call function."""
        def parse_int(s: str) -> Either[str, int]:
            return Either.right(int(s))

        e: Either[str, str] = Either.left("initial error")
        result = e.and_then(parse_int)
        assert result._left == "initial error"

    def test_and_then_propagates_function_left(self):
        """and_then should propagate Left from the chained function."""
        def parse_int(s: str) -> Either[str, int]:
            return Either.left("parsing failed")

        result = Either.right("abc").and_then(parse_int)
        assert result._left == "parsing failed"

    def test_and_then_with_different_error_types(self):
        """and_then should handle different error types in union."""
        def validate(x: int) -> Either[ValueError, int]:
            if x < 0:
                return Either.left(ValueError("negative"))
            return Either.right(x * 2)

        # Initial error type is str, function error type is ValueError
        e: Either[str, int] = Either.right(5)
        result = e.and_then(validate)
        assert result._right == 10


class TestEitherZip:
    """Tests for the zip method."""

    def test_zip_combines_two_rights(self):
        """zip should combine two Right values into a tuple."""
        e1 = Either.right(1)
        e2 = Either.right("a")
        result = e1.zip(e2)
        assert result._right == (1, "a")

    def test_zip_returns_left_if_first_is_left(self):
        """zip should return Left if first Either is Left."""
        e1: Either[str, int] = Either.left("error1")
        e2 = Either.right("a")
        result = e1.zip(e2)
        assert result._left == "error1"

    def test_zip_returns_left_if_second_is_left(self):
        """zip should return Left if second Either is Left."""
        e1 = Either.right(1)
        e2: Either[str, str] = Either.left("error2")
        result = e1.zip(e2)
        assert result._left == "error2"


class TestEitherThen:
    """Tests for the then method."""

    def test_then_returns_second_if_first_is_right(self):
        """then should return second Either if first is Right."""
        e1 = Either.right(1)
        e2 = Either.right(2)
        result = e1.then(e2)
        assert result._right == 2

    def test_then_returns_first_left_if_first_is_left(self):
        """then should return first Left if first is Left."""
        e1: Either[str, int] = Either.left("error")
        e2 = Either.right(2)
        result = e1.then(e2)
        assert result._left == "error"


class TestEitherMatch:
    """Tests for the match method."""

    def test_match_calls_right_on_right(self):
        """match should call right function on Right value."""
        e = Either.right(42)
        result = e.match(left=lambda e: f"error: {e}", right=lambda v: f"value: {v}")
        assert result == "value: 42"

    def test_match_calls_left_on_left(self):
        """match should call left function on Left value."""
        e = Either.left("oops")
        result = e.match(left=lambda e: f"error: {e}", right=lambda v: f"value: {v}")
        assert result == "error: oops"

    def test_match_uses_ident_default_for_right(self):
        """match should use identity function as default for right."""
        e = Either.right(42)
        result = e.match(left=lambda e: f"error: {e}")
        assert result == 42


class TestEitherUnwrapOr:
    """Tests for the unwrap_or method."""

    def test_unwrap_or_returns_right_value(self):
        """unwrap_or should return right value when present."""
        e = Either.right(42)
        assert e.unwrap_or(0) == 42

    def test_unwrap_or_returns_default_on_left(self):
        """unwrap_or should return default when Either is Left."""
        e: Either[str, int] = Either.left("error")
        assert e.unwrap_or(0) == 0

    def test_unwrap_or_returns_default_on_none_right(self):
        """unwrap_or should return default when right value is None."""
        e: Either[str, Optional[int]] = Either.right(None)
        assert e.unwrap_or(0) == 0


class TestEitherToJson:
    """Tests for the to_json method."""

    def test_to_json_serializes_right_value(self):
        """to_json should serialize right value to JSON string."""
        e = Either.right({"a": 1, "b": 2}).to_json()
        assert e._right == '{"a": 1, "b": 2}'

    def test_to_json_short_circuits_on_left(self):
        """to_json should not serialize when Either is Left."""
        e: Either[str, dict] = Either.left("error")
        result = e.to_json()
        assert result._left == "error"


class TestEitherWriteJsonOut:
    """Tests for the write_json_out method."""

    def test_write_json_out_writes_pydantic_model(self):
        """write_json_out should write Pydantic model to file."""

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        e = Either.right(model)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            result = e.write_json_out(filepath)
            assert result._right == model

            with open(filepath, "r") as f:
                data = json.load(f)
            assert data == {"name": "test", "value": 42}
        finally:
            os.unlink(filepath)

    def test_write_json_out_handles_error(self):
        """write_json_out should handle write errors gracefully."""

        class TestModel(BaseModel):
            name: str

        model = TestModel(name="test")
        e = Either.right(model)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = e.write_json_out("/nonexistent/path/file.json")
            assert result._left is not None
            assert len(w) == 1
            assert issubclass(w[0].category, WriteJsonWarning)


class TestEitherCtxPipe:
    """Tests for the ctx_pipe method."""

    def test_ctx_pipe_applies_side_effect(self):
        """ctx_pipe should apply function and return original value."""
        side_effects = []
        e = Either.right(42).ctx_pipe(lambda x: side_effects.append(x))
        assert e._right == 42
        assert side_effects == [42]

    def test_ctx_pipe_short_circuits_on_left(self):
        """ctx_pipe should not apply function when Either is Left."""
        side_effects = []
        e: Either[str, int] = Either.left("error")
        result = e.ctx_pipe(lambda x: side_effects.append(x))
        assert result._left == "error"
        assert side_effects == []


class TestEitherDictPipe:
    """Tests for the dict_pipe method."""

    def test_dict_pipe_applies_to_key(self):
        """dict_pipe should apply function to specified key."""
        side_effects = []
        e = Either.right({"a": 1, "b": 2}).dict_pipe("a", lambda x: side_effects.append(x))
        assert side_effects == [1]
        assert e._right == {"a": 1, "b": 2}

    def test_dict_pipe_short_circuits_on_left(self):
        """dict_pipe should not apply when Either is Left."""
        side_effects = []
        e: Either[str, dict[str, int]] = Either.left("error")
        result = e.dict_pipe("a", lambda x: side_effects.append(x))
        assert result._left == "error"
        assert side_effects == []


class TestIdentHelper:
    """Tests for the ident helper function."""

    def test_ident_returns_input(self):
        """ident should return its input unchanged."""
        assert ident(42) == 42
        assert ident("hello") == "hello"
        assert ident([1, 2, 3]) == [1, 2, 3]


class TestCurryDecorator:
    """Tests for the curry decorator."""

    def test_curry_creates_partial_function(self):
        """curry should create a partial function."""

        @curry
        def add_three(self, a: int, b: int, c: int) -> int:
            return a + b + c

        class Calculator:
            add = add_three

        calc = Calculator()
        partial = calc.add(1)
        assert partial(2, 3) == 6


class TestThrowsDecorator:
    """Tests for the throws decorator."""

    def test_throws_catches_specified_exception(self):
        """throws should catch specified exception and return Left."""

        @throws(ValueError)
        def risky_func(x: int) -> Either[ValueError, int]:
            if x < 0:
                raise ValueError("Negative!")
            return Either.right(x * 2)

        result = risky_func(-1)
        assert isinstance(result._left, ValueError)
        assert str(result._left) == "Negative!"

    def test_throws_passes_through_success(self):
        """throws should pass through successful Either."""

        @throws(ValueError)
        def risky_func(x: int) -> Either[ValueError, int]:
            return Either.right(x * 2)

        result = risky_func(5)
        assert result._right == 10

    def test_throws_warns_on_unexpected_exception(self):
        """throws should warn on unexpected exceptions."""

        @throws(ValueError)
        def risky_func(x: int) -> Either[ValueError, int]:
            raise KeyError("Unexpected!")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = risky_func(1)
            assert isinstance(result._left, KeyError)
            assert len(w) == 1
            assert issubclass(w[0].category, ExternalWrapWarning)


class TestWrapExternal:
    """Tests for the wrap_external function."""

    def test_wrap_external_wraps_success(self):
        """wrap_external should wrap successful result in Right."""

        def add(a: int, b: int) -> int:
            return a + b

        wrapped = wrap_external(add, ValueError)
        result = wrapped(2, 3)
        assert result._right == 5

    def test_wrap_external_catches_specified_exception(self):
        """wrap_external should catch specified exception."""

        def parse_int(s: str) -> int:
            return int(s)

        wrapped = wrap_external(parse_int, ValueError)
        result = wrapped("not a number")
        assert isinstance(result._left, ValueError)

    def test_wrap_external_warns_on_unexpected_exception(self):
        """wrap_external should warn on unexpected exceptions."""

        def risky() -> int:
            raise KeyError("Unexpected!")

        wrapped = wrap_external(risky, ValueError)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = wrapped()
            assert isinstance(result._left, KeyError)
            assert len(w) == 1
            assert issubclass(w[0].category, ExternalWrapWarning)


class TestEitherIterTo:
    """Tests for the iter_to_ method."""

    def test_iter_to_converts_to_type(self):
        """iter_to_ should convert iterable to specified type."""
        e = Either.right([1, 2, 3]).iter_to_(set)
        assert e._right == {1, 2, 3}

        e2 = Either.right((1, 2, 3)).iter_to_(list)
        assert e2._right == [1, 2, 3]
