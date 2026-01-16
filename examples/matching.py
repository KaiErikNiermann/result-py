from result_py.either import Either


# fmt: off
program1 = (
    Either.right(10)
    .match(
        right=lambda value: f"Value is: {value}",
        left=lambda error: f"Error occurred: {error}"
    )
)
# fmt: on

print(program1)  # Output: Value is: 10
