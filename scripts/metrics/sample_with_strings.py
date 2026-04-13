"""Small sample file to test string literal metrics."""

CONSTANT_STR = "hello world"
ANOTHER = "a longer string example with numbers 12345"


def f(x="default"):
    # inline comment
    s = "short"
    t = 'single-quoted'
    return s + t + x


class C:
    doc = "class docstring"
