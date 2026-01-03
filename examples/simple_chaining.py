from effect_py.either import Either
from typing import Callable

increment: Callable[[int], int] = lambda x: x + 1
double: Callable[[int], int] = lambda x: x * 2
subtract: Callable[[int], int] = lambda x: x - 3

# fmt: off
result = (
    Either.right(5)  
    .pipe(increment) # 6 
    .pipe(double)    # 12 
    .pipe(subtract)  # 9 
)
# fmt: on

print(result)  # Output: Either(_left=None, _right=9)
