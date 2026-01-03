from effect_py.either import Either
import asyncio


def apply_discount(total: float, discount_rate: float) -> Either[ValueError, float]:
    if discount_rate == 0:
        return Either.left(ValueError("Discount rate cannot be zero."))
    return Either.right(total * (1 - discount_rate))


async def main():
    transaction_amount = await asyncio.sleep(0, result=100)

    final_amount = (
        Either.right(transaction_amount)
        .pipe(lambda amt: amt * 2)
        .pipe(lambda amt: apply_discount(amt, 0.1))
    )

    return final_amount


if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)  # Output: Either(_left=None, _right=180.0)
