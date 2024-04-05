'''
Unit tests for one step of the bisction method
The function under test is wk06.wk06()

Would test the following:
1. the binary point is smaller than root
2. the binary point is larger than root
3. both lower end upper bounds are smaller than root
4. both lower end upper bounds are larger than root
'''

import math
import pathlib
import random
import sys

from typing import Callable


import pytest


file_path = pathlib.Path(__file__)
test_folder = file_path.parent.absolute()
proj_folder = test_folder.parent.absolute()


sys.path.insert(
    0,
    str(proj_folder)
)


import wk06


random.seed()


@pytest.fixture
def root() -> float:
    return random.uniform(-10, 10)


@pytest.fixture
def f(root:float) -> Callable[[float], float]:
    '''
    Returns an exponential function with a root at the given.
    '''
    a = random.uniform(2.0, 10.0)
    b = -a * math.exp(root)

    def return_this(x:float) -> float:
        return (a * math.exp(x)) + b

    assert math.isclose(return_this(root), 0.0)

    return return_this


@pytest.fixture
def small_epsilon() -> float:
    return 1e-6 * random.uniform(1.0, 9.0)


@pytest.fixture
def big_epsilon() -> float:
    return 1e-1 * random.uniform(1.0, 9.0)


def test__choose_upper__not_found(root:float, f:Callable[[float], float], small_epsilon:float):

    x_next = root + random.uniform(10.0, 5.0)
    assert x_next > root

    delta = random.uniform(20.0, 30.0)

    x_lower = x_next + (-delta)
    x_upper = x_next + delta

    assert math.isclose(abs(x_lower - x_next), abs(x_upper - x_next))
    assert (f(x_lower) * f(x_upper)) < 0

    d = wk06.wk06(f, x_lower, x_upper, epsilon=small_epsilon)

    assert not d['found']
    assert math.isclose(d['x_upper'], x_next)
    assert math.isclose(d['x_lower'], x_lower)


def test__choose_lower__not_found(root:float, f:Callable[[float], float], small_epsilon:float):

    x_next = root + random.uniform(-10.0, -5.0)
    assert x_next < root

    delta = random.uniform(20.0, 30.0)

    x_lower = x_next + (-delta)
    x_upper = x_next + delta

    assert math.isclose(abs(x_lower - x_next), abs(x_upper - x_next))
    assert (f(x_lower) * f(x_upper)) < 0

    d = wk06.wk06(f, x_lower, x_upper, epsilon=small_epsilon)

    assert not d['found']
    assert math.isclose(d['x_upper'], x_upper)
    assert math.isclose(d['x_lower'], x_next)


def test__choose_upper__found(root:float, f:Callable[[float], float], big_epsilon:float):

    x_next = root + random.uniform(0.05*big_epsilon, 0.1*big_epsilon)
    assert root < x_next

    delta = random.uniform(big_epsilon*0.1, big_epsilon*0.5)

    x_lower = x_next + (-delta)
    x_upper = x_next + delta

    assert x_next < x_upper
    assert x_lower < x_next

    assert root < x_upper
    assert x_lower < root

    assert math.isclose(abs(x_lower - x_next), abs(x_upper - x_next))
    assert (f(x_lower) * f(x_upper)) < 0

    d = wk06.wk06(f, x_lower, x_upper, epsilon=big_epsilon)

    assert d['found']
    assert math.isclose(d['x_upper'], x_next)
    assert math.isclose(d['x_lower'], x_lower)


def test__choose_lower__found(root:float, f:Callable[[float], float], big_epsilon:float):

    x_next = root + random.uniform((-0.1)*big_epsilon, (-0.05)*big_epsilon)
    assert x_next < root

    delta = random.uniform(big_epsilon*(0.1), big_epsilon*(0.5))

    x_lower = x_next + (-delta)
    x_upper = x_next + delta

    assert x_next < x_upper
    assert x_lower < x_next

    assert root < x_upper
    assert x_lower < root

    assert math.isclose(abs(x_lower - x_next), abs(x_upper - x_next))
    assert (f(x_lower) * f(x_upper)) < 0

    d = wk06.wk06(f, x_lower, x_upper, epsilon=big_epsilon)

    assert d['found']
    assert math.isclose(d['x_upper'], x_upper)
    assert math.isclose(d['x_lower'], x_next)

def test__both_below(root:float, f:Callable[[float], float], small_epsilon:float):
    x_lower = root + random.uniform(-10.0, -6.0)
    x_upper = root + random.uniform(-5.0, -1.0)

    assert x_lower < x_upper

    assert (f(x_lower) * f(x_upper)) > 0

    try:
        d = wk06.wk06(f, x_lower, x_upper, epsilon=small_epsilon)
    except ValueError:
        pass
    else:
        assert False, "Should have raised a ValueError"


def test__both_above(root:float, f:Callable[[float], float], small_epsilon:float):
    x_lower = root + random.uniform(1.0, 5.0)
    x_upper = root + random.uniform(6.0, 10.0)

    assert x_lower < x_upper

    assert (f(x_lower) * f(x_upper)) > 0

    try:
        d = wk06.wk06(f, x_lower, x_upper, epsilon=small_epsilon)
    except ValueError:
        pass
    else:
        assert False, "Should have raised a ValueError"


if "__main__" == __name__:
    pytest.main([__file__])
