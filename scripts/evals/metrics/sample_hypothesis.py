def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def factorial(num):
    if num <= 1:
        return 1
    return num * factorial(num-1)