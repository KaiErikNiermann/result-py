"""Comprehensive tests for the contracts module."""

import pytest
from typing import Mapping, Iterable
from result_py.contracts import HasItems, HasRootMapping, HasAdd, Identity


class TestIdentityType:
    """Tests for the Identity type alias."""

    def test_identity_is_passthrough(self):
        """Identity type should be a passthrough for any type."""
        x: Identity[int] = 42
        assert x == 42

        s: Identity[str] = "hello"
        assert s == "hello"

        lst: Identity[list[int]] = [1, 2, 3]
        assert lst == [1, 2, 3]


class TestHasItemsProtocol:
    """Tests for the HasItems protocol."""

    def test_dict_satisfies_has_items(self):
        """Regular dict should satisfy HasItems protocol."""
        d = {"a": 1, "b": 2}
        assert isinstance(d, HasItems)
        items = list(d.items())
        assert ("a", 1) in items
        assert ("b", 2) in items

    def test_custom_class_satisfies_has_items(self):
        """Custom class implementing items() should satisfy HasItems."""

        class CustomMapping:
            def items(self) -> Iterable[tuple[str, int]]:
                return [("x", 10), ("y", 20)]

        obj = CustomMapping()
        assert isinstance(obj, HasItems)
        assert list(obj.items()) == [("x", 10), ("y", 20)]

    def test_list_does_not_satisfy_has_items(self):
        """List should not satisfy HasItems protocol."""
        lst = [1, 2, 3]
        assert not isinstance(lst, HasItems)


class TestHasRootMappingProtocol:
    """Tests for the HasRootMapping protocol."""

    def test_custom_class_satisfies_has_root_mapping(self):
        """Custom class with root property should satisfy HasRootMapping."""

        class RootContainer:
            @property
            def root(self) -> Mapping[str, int]:
                return {"key1": 100, "key2": 200}

        obj = RootContainer()
        assert isinstance(obj, HasRootMapping)
        assert obj.root == {"key1": 100, "key2": 200}

    def test_plain_dict_does_not_satisfy_has_root_mapping(self):
        """Plain dict should not satisfy HasRootMapping."""
        d = {"root": {"key": 1}}
        assert not isinstance(d, HasRootMapping)


class TestHasAddProtocol:
    """Tests for the HasAdd protocol."""

    def test_int_satisfies_has_add(self):
        """int should satisfy HasAdd protocol."""
        x = 5
        assert isinstance(x, HasAdd)
        assert x + 3 == 8

    def test_str_satisfies_has_add(self):
        """str should satisfy HasAdd protocol."""
        s = "hello"
        assert isinstance(s, HasAdd)
        assert s + " world" == "hello world"

    def test_list_satisfies_has_add(self):
        """list should satisfy HasAdd protocol."""
        lst = [1, 2]
        assert isinstance(lst, HasAdd)
        assert lst + [3, 4] == [1, 2, 3, 4]

    def test_custom_class_satisfies_has_add(self):
        """Custom class implementing __add__ should satisfy HasAdd."""

        class Addable:
            def __init__(self, value: int):
                self.value = value

            def __add__(self, other: "Addable") -> "Addable":
                return Addable(self.value + other.value)

            def __eq__(self, other: object) -> bool:
                if isinstance(other, Addable):
                    return self.value == other.value
                return False

        a = Addable(10)
        b = Addable(20)
        assert isinstance(a, HasAdd)
        assert a + b == Addable(30)
