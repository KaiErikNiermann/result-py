from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Hashable,
    TypeVar,
    Generic,
    Iterable,
    Mapping,
    Optional,
    TypeIs,
    Union,
    cast,
    overload,
)
import json
from collections import Counter
import operator
from functools import reduce, wraps
import tqdm
from pydantic import BaseModel
import warnings

from result_py.contracts import HasAdd, HasRootMapping, Identity

T_co = TypeVar("T_co", covariant=True)


def ident[T](t: T) -> Identity[T]:
    return t


L = TypeVar("L")
R = TypeVar("R")


class ExternalWrapWarning(Warning):
    """Warning for unexpected exceptions in wrap_external."""


class WriteJsonWarning(Warning):
    """Warning for issues encountered while writing JSON files."""


@dataclass(frozen=True)
class Either(Generic[L, R]):
    """
    A type-safe Either monad for functional error handling.
    
    Either represents a value that can be one of two types:
    - Left (L): Typically used to represent an error or failure case
    - Right (R): Typically used to represent a success case
    
    This is a frozen (immutable) dataclass for safety.
    """

    _left: Optional[L] = None
    _right: Optional[R] = None

    @staticmethod
    def left[L2](l: L2) -> Either[L2, Any]:
        """Create a Left Either containing an error/failure value."""
        return Either(_left=l)

    # Alias for left
    failure = left

    @staticmethod
    def right[R2](r: R2) -> Either[Any, R2]:
        """Create a Right Either containing a success value."""
        return Either(_right=r)

    @staticmethod
    def _iter[T](
        iterable: Iterable[T],
        /,
        *,
        track: bool,
        f: Optional[Callable[..., Any]] = None,
        tqdm_kwargs: dict[str, Any],
    ) -> Iterable[T] | tqdm.tqdm[T]:
        if not track:
            return iterable

        if "desc" not in tqdm_kwargs and f is not None:
            tqdm_kwargs["desc"] = f.__name__

        return tqdm.tqdm(iterable, **tqdm_kwargs)

    # Alias for right
    success = right

    @property
    def is_right(self) -> bool:
        """Returns True if this Either contains a Right value."""
        return self._right is not None

    @property
    def is_left(self) -> bool:
        """Returns True if this Either contains a Left value."""
        return self._left is not None
    
    def _has_right(self, value: Optional[R]) -> TypeIs[R]:
        """Type guard helper for narrowing the right value."""
        return value is not None

    def _has_left(self, value: Optional[L]) -> TypeIs[L]:
        """Type guard helper for narrowing the left value."""
        return value is not None

    def _apply_right[T1, T2](
        self: Either[L, T1],
        f: Callable[[T1], T2],
    ) -> Either[L, T2]:
        """
        Applies a function to the right value of the Either if it exists.

        Parameters
        ----------
        self : `Either[L, T1]`
            An Either containing a right value of type T1.

        f : `Callable[[T1], T2]`
            A function that takes a value of type T1 and returns a value of type T2.

        Returns
        -------
        `Either[L, T2]`
            An Either containing the result of applying the function to the right value,
            or the left value if present.
        """
        if self._has_right(self._right):
            return Either.right(f(self._right))
        return Either.left(cast(L, self._left))

    def map_left[L2](
        self: Either[L, R],
        f: Callable[[L], L2],
    ) -> Either[L2, R]:
        """
        Applies a function to the left value of the Either if it exists.

        Parameters
        ----------
        f : `Callable[[L], L2]`
            A function that takes the left value and returns a new left value.

        Returns
        -------
        `Either[L2, R]`
            An Either containing the transformed left value, or the original right value.

        Example
        -------
        >>> e = Either.left("error")
        >>> e.map_left(lambda s: f"Error: {s}")
        >>> Either(_left="Error: error", _right=None)
        """
        if self._has_left(self._left):
            return Either.left(f(self._left))
        return Either.right(cast(R, self._right))

    def or_else[L2](
        self: Either[L, R],
        f: Callable[[L], Either[L2, R]],
    ) -> Either[L2, R]:
        """
        Applies a function to the left value to attempt recovery.

        Parameters
        ----------
        f : `Callable[[L], Either[L2, R]]`
            A function that takes the left value and returns a new Either,
            potentially recovering from the error.

        Returns
        -------
        `Either[L2, R]`
            The result of the recovery function if Left, otherwise the original Right.

        Example
        -------
        >>> def recover(err: str) -> Either[int, str]:
        ...     return Either.right("recovered") if err == "recoverable" else Either.left(500)
        >>> Either.left("recoverable").or_else(recover)
        >>> Either(_left=None, _right="recovered")
        """
        if self._has_left(self._left):
            return f(self._left)
        return Either.right(cast(R, self._right))

    @overload
    def pipe[*Ls, *Es, T](
        self: Either[Union[*Ls], R], f: Callable[[R], Either[Union[*Es], T]]
    ) -> Either[Union[*Ls, *Es], T]: ...

    @overload
    def pipe[*Ls, T](
        self: Either[Union[*Ls], R], f: Callable[[R], T]
    ) -> Either[Union[*Ls], T]: ...

    def pipe(self, f: Callable[[R], Any]) -> Either[Any, Any]:
        if self._has_right(self._right):
            out = f(self._right)
            if isinstance(out, Either):
                return out  # type: ignore
            return Either.right(out)
        return Either.left(cast(L, self._left))

    def to_json(self, **kwargs: Any) -> Either[L, str]:
        """
        Converts the Either to a JSON string representation.
        If the Either is left, it returns the left value as a JSON string.
        If the Either is right, it returns the right value as a JSON string.
        """
        if self._has_right(self._right):
            return Either.right(json.dumps(self._right, **kwargs))
        return Either.left(cast(L, self._left))

    def n_pipe[*T1, T2](
        self: Either[L, tuple[*T1]],
        f: Callable[[*T1], T2],
    ) -> Either[L, T2]:
        if self._has_right(self._right):
            return Either.right(f(*self._right))
        return Either.left(cast(L, self._left))

    @overload
    def _call[*T1, T2](self, f: Callable[[*T1], T2], arg: tuple[*T1]) -> T2: ...

    @overload
    def _call[T, T2](self, f: Callable[[T], T2], arg: T) -> T2: ...

    def _call[T2](self, f: Callable[..., T2], arg: Any) -> T2:
        if isinstance(arg, tuple):
            return f(*arg)
        else:
            return f(arg)

    @overload
    def partition[*T1](
        self: Either[L, Iterable[tuple[*T1]]],
        f: Callable[[*T1], bool],
    ) -> Either[L, tuple[Iterable[tuple[*T1]], Iterable[tuple[*T1]]]]: ...

    @overload
    def partition[T](
        self: Either[L, Iterable[T]],
        f: Callable[[T], bool],
    ) -> Either[L, tuple[Iterable[T], Iterable[T]]]: ...

    def partition(
        self: Either[L, Iterable[Any]],
        f: Callable[..., bool],
    ) -> Either[L, tuple[Iterable[Any], Iterable[Any]]]:
        def _partition(it: Iterable[Any]) -> tuple[list[Any], list[Any]]:
            truthy: list[Any] = []
            falsy: list[Any] = []
            for item in it:
                if self._call(f, item):
                    truthy.append(item)
                else:
                    falsy.append(item)
            return (truthy, falsy)
        return self._apply_right(_partition)

    def n_partition[*T1, *T2](
        self: Either[L, Iterable[Union[*T1, *T2]]],
        f: Callable[[Union[*T1, *T2]], TypeIs[Union[*T2]]],
    ) -> Either[L, tuple[Iterable[Union[*T1]], Iterable[Union[*T2]]]]:
        """
        Partitions the Iterable contained in the Either into two parts based on a predicate function.

        Parameters
        ----------
        self : `Either[L, Iterable[Union[*T1] | Union[*T2]]]`
            An Either containing an Iterable of items of type T1 or T2.

        f : `Callable[[Union[*T1] | Union[*T2]], TypeIs[T1]]`
            A function that takes an item of type T1 or T2 and returns True if the item is of type T1, False otherwise.

        Returns
        -------
        `Either[L, tuple[Iterable[*T1], Iterable[*T2]]]`
            An Either containing a tuple of two Iterables:
            - The first Iterable contains items of type T1.
            - The second Iterable contains items of type T2.
            If the Either is left, it returns the left value.
        """

        def _partition(it: Iterable[Union[*T1, *T2]]) -> tuple[list[Union[*T1]], list[Union[*T2]]]:
            first: list[Union[*T1]] = []
            second: list[Union[*T2]] = []
            for item in it:
                if f(item):
                    second.append(item)  # type: ignore
                else:
                    first.append(item)  # type: ignore
            return (first, second)

        return self._apply_right(_partition)

    def two_partition[T1, T2](
        self: Either[L, Iterable[T1 | T2]],
        f: Callable[[T1 | T2], TypeIs[T1]],
    ) -> Either[L, tuple[Iterable[T1], Iterable[T2]]]:
        """
        Partitions the Iterable contained in the Either into two parts based on a predicate function.

        Parameters
        ----------
        self : `Either[L, Iterable[T1 | T2]]`
            An Either containing an Iterable of items of type T1 or T2.

        f : `Callable[[T1 | T2], TypeIs[T1]]`
            A function that takes an item of type T1 or T2 and returns True if the item is of type T1, False otherwise.

        Returns
        -------
        `Either[L, tuple[Iterable[T1], Iterable[T2]]]`
            An Either containing a tuple of two Iterables:
            - The first Iterable contains items of type T1.
            - The second Iterable contains items of type T2.
            If the Either is left, it returns the left value.
        """

        def _partition(it: Iterable[T1 | T2]) -> tuple[list[T1], list[T2]]:
            t1_items: list[T1] = []
            t2_items: list[T2] = []
            for item in it:
                if f(item):
                    t1_items.append(item)  # type: ignore
                else:
                    t2_items.append(item)  # type: ignore
            return (t1_items, t2_items)

        return self._apply_right(_partition)

    def to_root_items[T](
        self: Either[L, HasRootMapping[T]],
    ) -> "Either[L, Iterable[tuple[str, T]]]":
        """
        Converts the Either containing a HasRootMapping to an Iterable of its items.

        Parameters
        ----------
        self : `Either[L, HasRootMapping[T]]`
            An Either containing a HasRootMapping with items of type T.

        Returns
        -------
        `Either[L, Iterable[tuple[str, T]]]`
            An Either containing an Iterable of tuples, where each tuple contains a string key and a value of type T.
        """
        return self._apply_right(lambda it: it.root.items())

    @overload
    def filter[*T1](
        self: Either[L, Iterable[tuple[*T1]]],
        f: Callable[[*T1], bool],
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, Iterable[tuple[*T1]]]: ...

    @overload
    def filter[T1](
        self: Either[L, Iterable[T1]],
        f: Callable[[T1], bool],
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, Iterable[T1]]: ...

    def filter(
        self: Either[L, Iterable[Any]],
        f: Callable[..., bool],
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, Iterable[Any]]:
        return self._apply_right(
            lambda it: filter(
                lambda item: self._call(f, item),
                self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs),
            )
        )

    def map[T1, T2](
        self: Either[L, Iterable[T1]],
        f: Callable[[T1], T2],
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, Iterable[T2]]:
        return self._apply_right(
            lambda it: map(
                f,
                self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs),
            )
        )

    def to_items[T1, T2](
        self: Either[L, Mapping[T1, T2]],
    ) -> Either[L, Iterable[tuple[T1, T2]]]:
        """
        Converts the Either containing a Mapping to an Iterable of its items.

        Parameters
        ----------
        self : `Either[L, Mapping[T1, T2]]`
            An Either containing a Mapping with items of type T1 and T2.

        Returns
        -------
        `Either[L, Iterable[tuple[T1, T2]]]`
            An Either containing an Iterable of tuples, where each tuple contains a key of type T1 and a value of type T2.
        """
        return self._apply_right(lambda it: it.items())

    def counted[T1](
        self: Either[L, Iterable[T1]],
    ) -> Either[L, Mapping[T1, int]]:
        """
        Counts the occurrences of each item in the Iterable contained in the Either.

        Parameters
        ----------
        self : `Either[L, Iterable[T1]]`
            An Either containing an Iterable of items of type T1.

        Returns
        -------
        `Either[L, Mapping[T1, int]]`
            An Either containing a Counter mapping each item to its count,
            or the left value if present.
        """
        return self._apply_right(Counter)

    def filter_map[T1, T2](
        self: Either[L, Iterable[T1]],
        f: Callable[[T1], Optional[T2]],
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, Iterable[T2]]:
        """
        Applies a filter-map function to each item in the Iterable contained in the Either.

        Parameters
        ----------
        self : `Either[L, Iterable[T1]]`
            An Either containing a Iterable of items of type T1.

        f : `Callable[[T1], Optional[T2]]`
            A function that takes an item of type T1 and returns an optional value of type T2.
            Items for which the function returns None are filtered out.

        track : `bool`, optional
            If True, uses tqdm to track progress.

        tqdm_kwargs : `dict[str, Any]`, optional
            Additional keyword arguments for tqdm.

        Returns
        -------
        `Either[L, Iterable[T2]]`
            An Either containing the results of applying the function to each item in the Iterable,
            excluding items where the function returns None.
            If the Either is left, it returns the left value.

        Example
        -------
        >>> e = Either.right([1, 2, 3, 4, 5])
        >>> e.filter_map(lambda x: x * 2 if x % 2 == 0 else None)
        >>> Either(_left=None, _right=[4, 8])
        """
        return self._apply_right(
            lambda it: [
                _item
                for item in (self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs))
                if (_item := f(item)) is not None
            ]
        )

    def flat_map[T1, T2](
        self: Either[L, Iterable[T1]],
        f: Callable[[T1], Iterable[T2]],
    ) -> Either[L, Iterable[T2]]:
        return self._apply_right(lambda it: (y for x in it for y in f(x)))

    def to_list[T1](self: Either[L, Iterable[T1]]) -> Either[L, list[T1]]:
        return self._apply_right(list)

    def n_map_reduce[*T1, T2](
        self: Either[L, Iterable[tuple[*T1]]],
        f: Callable[[*T1], HasAdd[T2]],
        initial: HasAdd[T2],
        add: Callable[[HasAdd[T2], HasAdd[T2]], HasAdd[T2]] = operator.add,
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, HasAdd[T2]]:
        """
        Maps each item in the Iterable contained in the Either to a new type using the provided function,
        and then reduces the resulting Iterable to a single value.

        Parameters
        ----------
        self : `Either[L, Iterable[tuple[*T1]]]`
            An Either containing an Iterable of tuples, where each tuple contains multiple items.

        f : `Callable[[...], T2]`
            A function that takes multiple arguments and returns a value of type T2.

        initial : `T2`
            The initial value for the reduction.

        Returns
        -------
        `Either[L, T2]`
            An Either containing the reduced value,
            or the left value if present.

        Example
        -------
        >>> e = Either.right([(1, 2), (3, 4), (5, 6)])
        >>> e.n_map_reduce(lambda x, y: x + y, 0)
        >>> Either(_left=None, _right=21)
        """
        return self._apply_right(
            lambda it: reduce(
                lambda acc, x: add(acc, f(*x)),
                self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs),
                initial,
            )
        )

    def iter_to_[T1, T2](self: Either[L, Iterable[T1]], t: type[T2]) -> Either[L, T2]:
        """
        Converts the Iterable contained in the Either to a specific type.

        Parameters
        ----------
        self : `Either[L, Iterable[T1]]`
            An Either containing an Iterable of items of type T1.

        t : `type[T2]`
            The type to which the items should be converted.

        Returns
        -------
        `Either[L, Iterable[T2]]`
            An Either containing an Iterable of items of type T2,
            or the left value if present.
        """
        return self._apply_right(t)

    def to_counter[T1: Hashable](
        self: Either[L, Iterable[T1]],
    ) -> Either[L, Mapping[T1, int]]:
        """
        Converts the Iterable contained in the Either to a Counter (a subclass of dict).

        Parameters
        ----------
        self : `Either[L, Iterable[T1]]`
            An Either containing an Iterable of items of type T1.

        Returns
        -------
        `Either[L, Mapping[T1, int]]`
            An Either containing a Counter mapping each item to its count,
            or the left value if present.
        """
        return self._apply_right(Counter)

    def map_reduce[T1, T2](
        self: Either[L, Iterable[T1]],
        f: Callable[[T1], HasAdd[T2]],
        initial: HasAdd[T2],
        add: Callable[[HasAdd[T2], HasAdd[T2]], HasAdd[T2]] = operator.add,
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, HasAdd[T2]]:
        return self._apply_right(
            lambda it: reduce(
                lambda acc, x: add(acc, f(x)),
                self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs),
                initial,
            )
        )

    def reduce[T1, T2](
        self: Either[L, Iterable[T1]],
        f: Callable[[T2, T1], T2],
        initial: T2,
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, T2]:
        """
        Reduces the Iterable contained in the Either to a single value using the provided function.

        Parameters
        ----------
        self : `Either[L, Iterable[T1]]`
            An Either containing an Iterable of items of type T1.

        f : `Callable[[T2, T1], T2]`
            A function that takes an accumulator of type T2 and an item of type T1,
            and returns a new accumulator of type T2.

        initial : `T2`
            The initial value for the reduction.

        Returns
        -------
        `Either[L, T2]`
            An Either containing the reduced value,
            or the left value if present.

        Example
        -------
        >>> e = Either.right([1, 2, 3, 4])
        >>> e.reduce(lambda acc, x: acc + x, 0)
        >>> Either(_left=None, _right=10)
        """
        return self._apply_right(
            lambda it: reduce(
                f,
                self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs),
                initial,
            )
        )

    def to_set[T1](self: Either[L, Iterable[T1]]) -> Either[L, set[T1]]:
        """
        Converts the Iterable contained in the Either to a set.

        Parameters
        ----------
        self : `Either[L, Iterable[T1]]`
            An Either containing an Iterable of items of type T1.

        Returns
        -------
        `Either[L, set[T1]]`
            An Either containing a set of items from the Iterable,
            or the left value if present.
        """
        return self._apply_right(set)

    def flatten[T1](self: Either[L, Iterable[Iterable[T1]]]) -> Either[L, Iterable[T1]]:
        """
        Flattens an Either containing an iterable of iterables into an iterable of items.

        Parameters
        ----------
        self : `Either[L, Iterable[Iterable[T1]]]`
            An Either containing an iterable of iterables to be flattened.

        Returns
        -------
        `Either[L, Iterable[T1]]`
            An Either containing a single iterable with all items from the sub-iterables,
            or the left value if present.
        """
        if self._has_right(self._right):
            return Either.right((item for sublist in self._right for item in sublist))
        return Either.left(cast(L, self._left))

    def n_filter_map[*T1, T2](
        self: Either[L, Iterable[tuple[*T1]]],
        f: Callable[[*T1], Optional[T2]],
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, Iterable[T2]]:
        """
        Applies an *unpacking* filter-map function to each item in the Iterable contained in the Either.

        Parameters
        ----------
        self : `Either[L, Iterable[tuple[*T1]]]`
            An Either containing an Iterable of tuples, where each tuple contains multiple items.

        f : `Callable[[...], Optional[T2]]`
            A function that takes multiple arguments and returns an optional value of type T2.
            Items for which the function returns None are filtered out.

        track : `bool`, optional
            If True, uses tqdm to track progress.

        tqdm_kwargs : `dict[str, Any]`, optional
            Additional keyword arguments for tqdm.

        Returns
        -------
        `Either[L, Iterable[T2]]`
            An Either containing the results of applying the function to each item in the Iterable,
            excluding items where the function returns None.
            If the Either is left, it returns the left value.

        Example
        -------
        >>> e = Either.right([(1, 2), (3, 4), (5, 6)])
        >>> e.n_filter_map(lambda x, y: x + y if (x + y) > 5 else None)
        >>> Either(_left=None, _right=[7, 11])
        """
        def _gen(it: Iterable[tuple[*T1]]) -> Iterable[T2]:
            for item in self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs):
                if (_item := f(*item)) is not None:
                    yield _item
        return self._apply_right(_gen)

    def n_map[*T1, T2](
        self: Either[L, Iterable[tuple[*T1]]],
        f: Callable[[*T1], T2],
        track: bool = False,
        **tqdm_kwargs: dict[str, Any],
    ) -> Either[L, Iterable[T2]]:
        """
        Applies an *unpacking* function to each item in the Iterable contained in the Either.

        Parameters
        ----------
        f : `Callable[[...], T2]`
            A function that takes multiple arguments and returns a value of type T2.

        track : `bool`, optional
            If True, uses tqdm to track progress.

        tqdm_kwargs : `dict[str, Any]`, optional
            Additional keyword arguments for tqdm.

        Returns
        -------
        `Either[L, Iterable[T2]]`
            An Either containing the results of applying the function to each item in the Iterable.
            If the Either is left, it returns the left value.

        Example
        -------
        >>> e = Either.right([(1, 2), (3, 4)])
        >>> e.n_map(lambda x, y: x + y)
        >>> Either(_left=None, _right=[3, 7])
        """
        def _gen(it: Iterable[tuple[*T1]]) -> Iterable[T2]:
            for item in self._iter(it, track=track, tqdm_kwargs=tqdm_kwargs):
                yield f(*item)
        return self._apply_right(_gen)

    def dict_pipe(
        self: Either[L, Mapping[str, T_co]], key: str, f: Callable[[T_co], Any]
    ) -> Either[L, Mapping[str, T_co]]:
        if self._has_right(self._right) and key in self._right:
            f(self._right[key])
            return Either.right(self._right)
        return Either.left(cast(L, self._left))

    def ctx_pipe[T1, T2](self: Either[T1, R], f: Callable[[R], T2]) -> Either[T1, R]:
        """
        Applies a function to the right value of the Either, discards the return value and passes along the right value.

        Parameters
        ----------
        f : `Callable[[R], Either[T1, T2]]`
            A function that takes the right value and returns an Either.

        Returns
        -------
        `Either[T1, R]`
            An Either containing the left value from the function if it fails, or the original right value if it succeeds.

        Example
        -------
        >>> e = Either.right(5)
        >>> e.ctx_pipe(lambda x: logger.info(f"Value: {x}"))
        >>> Either(_left="Error", _right=5)
        """
        if self._has_right(self._right):
            f(self._right)
            return Either.right(self._right)
        return Either.left(cast(T1, self._left))

    def unwrap_or(self, default: R) -> R:
        """
        Returns the right value if present, otherwise returns the default.

        Parameters
        ----------
        default : `R`
            The default value to return if this Either is Left.

        Returns
        -------
        `R`
            The right value if present, otherwise the default.

        Example
        -------
        >>> Either.right(42).unwrap_or(0)
        42
        >>> Either.left("error").unwrap_or(0)
        0
        """
        if self._has_right(self._right):
            return self._right
        return default

    def and_then[L2, R2](
        self: Either[L, R],
        f: Callable[[R], Either[L2, R2]],
    ) -> Either[L | L2, R2]:
        """
        Chains an Either-returning function, similar to Rust's and_then.

        This is the monadic bind operation. If this Either is Right, applies the function
        to the right value and returns the result. If this Either is Left, returns the Left.

        Parameters
        ----------
        f : `Callable[[R], Either[L2, R2]]`
            A function that takes the right value and returns a new Either.

        Returns
        -------
        `Either[L | L2, R2]`
            The result of applying f if Right, otherwise the original Left.

        Example
        -------
        >>> def parse_int(s: str) -> Either[str, int]:
        ...     try:
        ...         return Either.right(int(s))
        ...     except ValueError:
        ...         return Either.left("not a number")
        >>> Either.right("42").and_then(parse_int)
        >>> Either(_left=None, _right=42)
        """
        if self._has_right(self._right):
            # Cast is safe: L2 is a subset of L | L2
            return cast(Either[L | L2, R2], f(self._right))
        # Cast is safe: L is a subset of L | L2
        return cast(Either[L | L2, R2], Either.left(cast(L, self._left)))

    def zip[B](self: "Either[L, R]", other: "Either[L, B]") -> "Either[L, tuple[R, B]]":
        """
        Combines two Either instances into a single Either containing a tuple of their right values if present.

        Parameters
        ----------
        other : `Either[L, B]`
            Another Either instance to zip with.

        Returns
        -------
        `Either[L, tuple[R, B]]`
            An Either containing a tuple of right values if both are Right, otherwise the first encountered Left.
        """
        return Either(
            _left=self._left if self._has_left(self._left) else other._left,
            _right=(
                (self._right, other._right)
                if self._has_right(self._right) and other._has_right(other._right)
                else None
            ),
        )

    def write_json_out[T1: BaseModel](self: Either[L, T1], fp: str) -> Either[L, T1]:
        """
        Writes the right value of the Either to a JSON file if it is a BaseModel.

        Parameters
        ----------
        self: `Either[L, T1]`
            An Either containing a BaseModel instance as its right value.

        fp : `str`
            The file path where the JSON representation of the BaseModel will be written.

        Returns
        -------
        `Either[L, None]`
            An Either that is Right with None if the write operation is successful,
            or Left with the original left value if the Either is Left or if an error occurs during
            the write operation.

        """
        if self._has_right(self._right):  #  type: ignore
            try:
                with open(fp, "w") as f:
                    f.write(self._right.model_dump_json(indent=4))  # type: ignore
                return self
            except OSError as e:
                warnings.warn(
                    f"Failed to write JSON to {fp}: {e}",
                    category=WriteJsonWarning,
                    stacklevel=2,
                )
                return Either.left(cast(L, e))
        return Either.left(cast(L, self._left))

    def then[T1, T2](self, mnd: Either[T1, T2]) -> Either[T1, T2]:
        """
        Chains another Either operation, returning the new Either if the current one is Right,
        or the current Left value if it is Left.

        Parameters
        ----------
        mnd : `Either[L, T]`
            An Either to chain with the current Either.

        Returns
        -------
        `Either[L, T]`
            The new Either if the current one is Right, otherwise the current Left value.

        Example
        -------
        Good for chaining operations that depend on the success of the previous one.
        >>> e1 = Either.right(5)
        >>> e2 = Either.right(10)
        >>> e1.then(e2)
        >>> Either(_left=None, _right=10)
        """
        if self._has_right(self._right):
            return mnd
        return Either.left(cast(T1, self._left))

    def match[Out1, Out2](
        self, left: Callable[[L], Out1], right: Callable[[R], Out2] = ident
    ) -> Out1 | Out2:
        """
        Matches the Either to either the left or right function based on its state.

        Parameters
        ----------
        left : `Callable[[L], Out1]`
            A function to apply if the Either is Left, taking the left value as an argument.

        right : `Callable[[R], Out2]`, optional
            A function to apply if the Either is Right, taking the right value as an argument.

        Returns
        -------
        `Out1 | Out2`
            The result of applying the left function if the Either is Left,
            or the right function if the Either is Right.
        """
        if self._has_right(self._right):
            return right(self._right)
        return left(cast(L, self._left))


def as_either[T](value: T) -> Either[None, T]:
    return Either.right(value)


def as_failure[L](value: L) -> Either[L, None]:
    return Either.left(value)


def curry[S, T, *Ts, R](
    func: Callable[[S, T, *Ts], R],
) -> Callable[[S, T], Callable[[*Ts], R]]:
    """
    Curries a function that takes at least one argument of type T and any number of additional arguments of type Ts.

    Parameters
    ----------
    func : `Callable[[T, *Ts], R]`
        The function to be curried.

    Returns
    -------
    `Callable[[T], Callable[[*Ts], R]]`
        A curried version of the function that takes the first argument and returns a new function
        that takes the remaining arguments.
    """

    @wraps(func)
    def decorator(self: S, t: T) -> Callable[[*Ts], R]:
        @wraps(func)
        def wrapper(*args: *Ts) -> R:
            return func(self, t, *args)

        return wrapper

    return decorator


def throws[T, *E1s, *E2s, **Ps](
    *exception_types: *E2s,
) -> Callable[
    [Callable[Ps, Either[Union[*E1s], T]]], Callable[Ps, Either[Union[*E1s, *E2s], T]]
]:
    """
    Decorator that wraps a function to catch specified exceptions and return them as Either.Left.

    Parameters
    ----------
    func : `Callable[Ps, Either[Union[*E1s], T]]`
        The function to be wrapped. It should return an Either containing either a left value of type Union[*E1s] or a right value of type T.

    Returns
    -------
    `Callable[[*E2s, *Ps], Callable[Ps, Either[Union[*E1s, *E2s], T]]]`
        A wrapped function that catches specified exceptions and returns them as Either.Left.

    Example
    -------
    >>> @throws(ValueError, KeyError)
    ... def risky_function(x: int) -> Either[Union[ValueError, KeyError], int]:
    ...     if x < 0:
    ...         raise ValueError("Negative value!")
    ...     elif x == 0:
    ...         raise KeyError("Zero value!")
    ...     return Either.right(x * 2)
    >>> result = risky_function(-1)
    >>> result
    Either(_left=ValueError('Negative value!'), _right=None)
    """

    def decorator(
        fn: Callable[Ps, Either[Union[*E1s], T]],
    ) -> Callable[Ps, Either[Union[*E1s, *E2s], T]]:

        @wraps(fn)
        def wrapper(
            *args: Ps.args, **kwargs: Ps.kwargs
        ) -> Either[Union[*E1s, *E2s], T]:
            try:
                # Cast is safe: E1s is a subset of E1s | E2s
                return cast(Either[Union[*E1s, *E2s], T], fn(*args, **kwargs))
            except exception_types as e:  # type: ignore
                # Cast is safe because its just generalizing (E1, E2, ..., En) as e: Union[E1, E2, ..., En]
                return Either.left(cast(Union[*E1s, *E2s], e))
            except Exception as e:
                warnings.warn(
                    f"An exception occured which was not statically specified in the exception_types: '{e}'.",
                    category=ExternalWrapWarning,
                    stacklevel=2,
                )
                return Either.left(e)  # type: ignore

        return wrapper

    return decorator


def wrap_external[T, *Ts, **Ps](
    func: Callable[Ps, T], *exception_types: *Ts
) -> Callable[Ps, Either[Union[*Ts], T]]:
    @wraps(func)
    def wrapper(*args: Ps.args, **kwargs: Ps.kwargs) -> Either[Union[*Ts], T]:
        try:
            result = func(*args, **kwargs)
            return Either.right(result)
        except exception_types as e:  # type: ignore
            # Cast is safe because its just generalizing (E1, E2, ..., En) as e: Union[E1, E2, ..., En]
            return Either.left(cast(Union[*Ts], e))
        except Exception as e:
            warnings.warn(
                f"An exception occured which was not statically specified in the exception_types: '{e}'.",
                category=ExternalWrapWarning,
                stacklevel=2,
            )
            return Either.left(e)  # type: ignore

    return wrapper
