"""Integration tests based on examples from the examples directory."""

import pytest
import asyncio
from result_py.either import Either


class TestApplyDiscountExample:
    """Tests based on examples/apply_discount.py"""

    def test_apply_discount_success(self):
        """Test applying a valid discount."""

        def apply_discount(
            total: float, discount_rate: float
        ) -> Either[ValueError, float]:
            if discount_rate == 0:
                return Either.left(ValueError("Discount rate cannot be zero."))
            return Either.right(total * (1 - discount_rate))

        result = apply_discount(100.0, 0.1)
        assert result._right == 90.0

    def test_apply_discount_zero_rate(self):
        """Test applying a zero discount rate."""

        def apply_discount(
            total: float, discount_rate: float
        ) -> Either[ValueError, float]:
            if discount_rate == 0:
                return Either.left(ValueError("Discount rate cannot be zero."))
            return Either.right(total * (1 - discount_rate))

        result = apply_discount(100.0, 0.0)
        assert isinstance(result._left, ValueError)
        assert str(result._left) == "Discount rate cannot be zero."

    @pytest.mark.asyncio
    async def test_async_discount_pipeline(self):
        """Test async pipeline with discount."""

        def apply_discount(
            total: float, discount_rate: float
        ) -> Either[ValueError, float]:
            if discount_rate == 0:
                return Either.left(ValueError("Discount rate cannot be zero."))
            return Either.right(total * (1 - discount_rate))

        transaction_amount = await asyncio.sleep(0, result=100)

        final_amount = (
            Either.right(transaction_amount)
            .pipe(lambda amt: amt * 2)
            .pipe(lambda amt: apply_discount(amt, 0.1))
        )

        assert final_amount._right == 180.0


class TestMatchingExample:
    """Tests based on examples/matching.py"""

    def test_match_with_right_value(self):
        """Test matching on a right value."""
        result = Either.right(10).match(
            right=lambda value: f"Value is: {value}",
            left=lambda error: f"Error occurred: {error}",
        )
        assert result == "Value is: 10"

    def test_match_with_left_value(self):
        """Test matching on a left value."""
        result = Either.left("Something went wrong").match(
            right=lambda value: f"Value is: {value}",
            left=lambda error: f"Error occurred: {error}",
        )
        assert result == "Error occurred: Something went wrong"


class TestSimpleChainingExample:
    """Tests based on examples/simple_chaining.py"""

    def test_chaining_operations(self):
        """Test chaining multiple operations."""
        increment = lambda x: x + 1
        double = lambda x: x * 2
        subtract = lambda x: x - 3

        result = (
            Either.right(5)
            .pipe(increment)  # 6
            .pipe(double)  # 12
            .pipe(subtract)  # 9
        )

        assert result._right == 9

    def test_chaining_with_initial_left(self):
        """Test that chaining short-circuits on left."""
        increment = lambda x: x + 1
        double = lambda x: x * 2

        result: Either[str, int] = (
            Either.left("Initial error").pipe(increment).pipe(double)
        )

        assert result._left == "Initial error"


class TestReadmeExamples:
    """Tests based on examples from README.md"""

    def test_basic_usage(self):
        """Test basic usage from README."""
        success = Either.right(42)
        failure = Either.left("Error")

        assert success._right == 42
        assert failure._left == "Error"

    def test_basic_chaining(self):
        """Test basic chaining from README."""
        result = Either.right(10).pipe(lambda x: x * 2).pipe(lambda x: x + 5)
        assert result._right == 25

    def test_error_handling_pipeline(self):
        """Test error handling pipeline from README."""

        def divide(a: float, b: float) -> Either[str, float]:
            if b == 0:
                return Either.left("Division by zero!")
            return Either.right(a / b)

        def sqrt(x: float) -> Either[str, float]:
            if x < 0:
                return Either.left("Cannot take sqrt of negative number!")
            return Either.right(x**0.5)

        result = (
            Either.right(16.0)
            .pipe(lambda x: divide(x, 2))  # Right(8.0)
            .pipe(sqrt)  # Right(2.83...)
        )

        assert result._right == pytest.approx(2.828, rel=0.01)

    def test_error_handling_pipeline_with_error(self):
        """Test error handling pipeline that results in error."""

        def divide(a: float, b: float) -> Either[str, float]:
            if b == 0:
                return Either.left("Division by zero!")
            return Either.right(a / b)

        def sqrt(x: float) -> Either[str, float]:
            if x < 0:
                return Either.left("Cannot take sqrt of negative number!")
            return Either.right(x**0.5)

        result = Either.right(16.0).pipe(lambda x: divide(x, 0))

        assert result._left == "Division by zero!"

    def test_collection_operations(self):
        """Test collection operations from README."""
        result = (
            Either.right([1, 2, 3, 4, 5])
            .filter(lambda x: x % 2 == 0)
            .map(lambda x: x * 10)
            .to_list()
        )
        assert result._right == [20, 40]

    def test_filter_map_operations(self):
        """Test filter_map from README."""
        result = (
            Either.right([1, 2, 3, 4, 5])
            .filter_map(lambda x: x * 2 if x > 2 else None)
            .to_list()
        )
        assert list(result._right) == [6, 8, 10]

    def test_match_usage(self):
        """Test match usage from README."""
        success_result = Either.right(8.0)
        message = success_result.match(
            left=lambda err: f"Failed: {err}", right=lambda val: f"Result: {val:.2f}"
        )
        assert message == "Result: 8.00"

        failure_result = Either.left("Division by zero!")
        message = failure_result.match(
            left=lambda err: f"Failed: {err}", right=lambda val: f"Result: {val:.2f}"
        )
        assert message == "Failed: Division by zero!"


class TestEdgeCases:
    """Tests for edge cases and potential bugs."""

    def test_empty_iterable_operations(self):
        """Test operations on empty iterables."""
        result = Either.right([]).filter(lambda x: x > 0).map(lambda x: x * 2).to_list()
        assert result._right == []

    def test_single_element_iterable(self):
        """Test operations on single-element iterables."""
        result = (
            Either.right([42]).filter(lambda x: x > 0).map(lambda x: x * 2).to_list()
        )
        assert result._right == [84]

    def test_none_as_right_value(self):
        """Test None as a right value."""
        e: Either[str, None] = Either.right(None)
        # is_right returns False for None, which is the expected behavior
        assert e._right is None
        assert e._left is None

    def test_complex_nested_pipeline(self):
        """Test complex nested pipeline."""

        def validate_positive(x: int) -> Either[str, int]:
            return Either.right(x) if x > 0 else Either.left("Not positive")

        def validate_even(x: int) -> Either[str, int]:
            return Either.right(x) if x % 2 == 0 else Either.left("Not even")

        result = (
            Either.right(4)
            .pipe(validate_positive)
            .pipe(validate_even)
            .pipe(lambda x: x * 2)
        )
        assert result._right == 8

        result_fail = (
            Either.right(3)
            .pipe(validate_positive)
            .pipe(validate_even)
            .pipe(lambda x: x * 2)
        )
        assert result_fail._left == "Not even"

    def test_deeply_nested_flatten(self):
        """Test flattening deeply nested structures (one level)."""
        result = Either.right([[1, 2], [3, 4], [5, 6]]).flatten().to_list()
        assert result._right == [1, 2, 3, 4, 5, 6]

    def test_reduce_with_empty_iterable(self):
        """Test reduce with empty iterable returns initial value."""
        result = Either.right([]).map_reduce(lambda x: x, 0)
        assert result._right == 0

    def test_counter_with_hashable_items(self):
        """Test counter with various hashable items."""
        result = Either.right([1, "a", 1, "a", 2, "b"]).to_counter()
        assert result._right == {1: 2, "a": 2, 2: 1, "b": 1}
