from math import *
from copy import deepcopy
import time


def bisection_method(f, a, b, tol=1e-6):
    """
    Find the root of the equation f(x) = 0 using the bisection method.

    Args:
        f: The target function, f(x) = 0.
        a: The left endpoint of the interval.
        b: The right endpoint of the interval.
        tol: The tolerance, stop when |f(mid)| < tol.

    Returns:
        The approximate root x.
    """
    
    fl = f(a)
    fr = f(b)
    l = deepcopy(a)
    r = deepcopy(b)
    fmid = f((l + r) / 2)
    if fl * fr > 0:
        raise ValueError('Both endpoints have the same sign.')
    
    while abs(fmid) > tol:
        if fl * fmid <= 0:
            fr = fmid
            r = (l + r) / 2
        else:
            fl = fmid
            l = (l + r) / 2
        
        fmid = f((l + r) / 2)
    
    return (l + r) / 2


def newton_method(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton method solving f(x) = 0.

    Args:
        f: objective function.
        df: derivative of the objective function.
        x0: initial value.
        tol: if |f(x)| < tol, stop.
        max_iter: maximum iteration number.

    reture:
        approximate solution x.
    """

    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(fx) <= tol:
            return x
        
        if dfx == 0:
            raise ValueError('Derivative is zero.')
        
        x = x - fx / dfx
    return x

# Algorithm testing
if __name__ == '__main__':
    # Define the function and its derivative
    def f(x):
        return x ** 2 - 2
    
    def df(x):
        return 2 * x
    
    # Initial parameters
    a, b = 1, 2
    x0 = 2
    tol = 1e-8
    exact_solution = sqrt(2)
    
    # Measure time and solve using Newton's method
    start_time_newton = time.time()
    x_sol_newton = newton_method(f, df, x0, tol)
    newton_time = time.time() - start_time_newton
    newton_error = abs(exact_solution - x_sol_newton)
    
    # Measure time and solve using Bisection method
    start_time_bisection = time.time()
    x_sol_bisection = bisection_method(f, a, b, tol)
    bisection_time = time.time() - start_time_bisection
    bisection_error = abs(exact_solution - x_sol_bisection)
    
    # Print results in a table
    print(f"{'Method':<15} {'Solution':<15} {'Error':<15} {'Time (s)':<15}")
    print('-' * 60)
    print(f"{'Newton':<15} {x_sol_newton:<15.10f} {newton_error:<15.3e} {newton_time:<15.3e}")
    print(f"{'Bisection':<15} {x_sol_bisection:<15.10f} {bisection_error:<15.3e} {bisection_time:<15.3e}")
    print('-' * 60)
    print(f"The exact solution: {exact_solution:.6f}")
