from ch04.numericalGradient import numerical_gradient_2d


def tangent_line(f, x):
    d = numerical_gradient_2d(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y
