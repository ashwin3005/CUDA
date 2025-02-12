def factorial(n: int):
    # n must be a positive integer
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
    

Eulers_constant = 0
for n in range(0, 10):
    Eulers_constant += (1/factorial(n))

print(Eulers_constant)  # 2.7182818...


# Euler's constant (e) is an irrational number. The larger the value of n, more precise the value will be.