from langchain.agents import tool


@tool
def is_prime(num: int) -> bool:
    """
    Returns True if the given number is prime, False otherwise
    :param num: integer number
    :return: bool indicating if num is prime
    """

    if not isinstance(num, int):
        num = int(num)

    if num < 2:     # 1 is not prime and negative numbers can't be prime
        return False
    for i in range(2, int(num ** 0.5) + 1):  # check for factors up to the square root of num
        if num % i == 0:
            return False
    return True

