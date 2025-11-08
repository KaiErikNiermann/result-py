from typing import Iterable, Mapping, Protocol, TypeVar, runtime_checkable

type Identity[X] = X

T_co = TypeVar("T_co", covariant=True)

T_add = TypeVar("T_add")


@runtime_checkable
class HasItems[K, V](Protocol):
    def items(self) -> Iterable[tuple[K, V]]: ...


@runtime_checkable
class HasRootMapping(Protocol[T_co]):
    @property
    def root(self) -> Mapping[str, T_co]: ...


@runtime_checkable
class HasAdd(Protocol[T_add]):
    def __add__(self: T_add, other: T_add) -> T_add: ...
