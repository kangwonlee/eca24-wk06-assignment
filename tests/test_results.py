'''
Unit tests for one step of the bisction method
The function under test are : 
+ wk06.wk06_cost()
+ wk06.wk06_curve()

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

from typing import Callable, Tuple


import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import numpy.testing as nt
import scipy.signal as ss
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
nr.seed()


'''
x(t) = A * exp(-zeta * w * t) * sin(w_d * t + phi)

x(0) = x0
v(0) = v0



'''

@ pytest.fixture
def m_kg() -> float:
    return random.uniform(1.0, 10.0)


@pytest.fixture(params=[0.2, 0.9])
def zeta(request) -> float:
    return request.param + random.uniform(-0.01, 0.01)


@pytest.fixture
def omega_rad() -> float:
    return random.uniform(1, 10)


@pytest.fixture
def DC() -> float:
    return random.uniform(-5, 5)


@pytest.fixture
def sampling_hz() -> float:
    return random.randint(1, 100) * 1000


@pytest.fixture
def k_Nm(m_kg:float, omega_rad:float) -> float:
    return m_kg * omega_rad**2


@pytest.fixture
def c_Nms(m_kg:float, zeta:float, omega_rad:float) -> float:
    '''
    c/m  = 2 zeta omega
    '''
    return 2 * zeta * omega_rad * m_kg


@pytest.fixture
def x_0_m() -> float:
    return random.uniform(0.5, 1.0)


@pytest.fixture
def v_0_mps() -> float:
    return random.uniform(-0.5, 0.0)


@pytest.fixture
def C1(m_kg, c_Nms, k_Nm, x_0_m, v_0_mps) -> float:

    assert m_kg is not None
    assert c_Nms is not None
    assert k_Nm is not None
    assert x_0_m is not None
    assert v_0_mps is not None

    return (
        c_Nms*x_0_m/(2*math.sqrt(k_Nm*m_kg)*math.sqrt(-c_Nms**2/(4*k_Nm*m_kg) + 1.0))
        + v_0_mps/(math.sqrt(k_Nm/m_kg)*math.sqrt(-c_Nms**2/(4*k_Nm*m_kg) + 1.0))
    )


@pytest.fixture
def C2(x_0_m) -> float:
    return (x_0_m)


@pytest.fixture
def A(C1, C2):
    return (C1**2 + C2**2)**0.5


@pytest.fixture
def phi_rad(C1, C2):
    return math.atan2(C2, C1)


@pytest.fixture
def ss_model(m_kg:float, c_Nms:float, k_Nm:float) -> ss.lti:
    mA = np.array([[0, 1.0], [-k_Nm/m_kg, -c_Nms/m_kg]])
    mB = np.array([[0.0], [1.0/m_kg]])
    mC = np.array([[1.0, 0.0]])
    mD = np.array([[0.0]])

    return ss.StateSpace(mA, mB, mC, mD)


@pytest.fixture
def t0_sec() -> float:
    return 0.0


@pytest.fixture
def te_sec() -> float:
    return 10.0 + random.uniform(0.1, 0.9)


@pytest.fixture
def n_sample(t0_sec:float, te_sec:float, sampling_hz:float):
    return int((te_sec - t0_sec) * sampling_hz) + 1


@pytest.fixture
def t_sec(t0_sec:float, te_sec:float, n_sample:float):
    return np.linspace(t0_sec, te_sec, n_sample)


@pytest.fixture(params=[0.1, 0.9])
def stdev(request) -> float:
    return request.param + random.uniform(-0.01, 0.01)


@pytest.fixture
def noise(t_sec:float, stdev:float) -> np.ndarray:
    return nr.normal(0.0, stdev, t_sec.shape)


@pytest.fixture
def t_x_dc(
        ss_model:ss.lti, t_sec:float, DC:float,
        x_0_m:float, v_0_mps:float
    ) -> Tuple[np.ndarray]:
    u_N = np.zeros_like(t_sec)

    x0_array = np.array([x_0_m, v_0_mps])

    t_sec, x_m, q = ss.lsim(
        ss_model, u_N, t_sec,
        X0=x0_array
    )

    x_m += DC

    return t_sec, x_m


@pytest.fixture
def t_x(
        t_x_dc:Tuple[np.ndarray],
        noise:np.ndarray
    ) -> Tuple[np.ndarray]:
    x_contaminated_m = np.array(t_x_dc[1]) + noise

    return t_x_dc[0], x_contaminated_m


@pytest.fixture
def param(A:float, zeta:float, omega_rad:float, phi_rad:float, DC:float) -> np.ndarray:
    return np.array((A, zeta, omega_rad, phi_rad, DC))


@pytest.fixture
def result_cost(param:np.ndarray, t_x:Tuple[np.ndarray]) -> float:
    return wk06.wk06_cost(param, *t_x)


@pytest.fixture
def result_curve(param:np.ndarray, t_x:Tuple[np.ndarray]) -> float:
    return wk06.wk06_curve(*param, t_x[0])


def compare_plot(t_array, x_measure, x_sim, x_result, png_filename, title=''):
    plt.clf()
    plt.plot(t_array, x_measure, 'o-', label='measurement')
    plt.plot(t_array, x_sim, 'o-', label='simulated')
    plt.plot(t_array, x_result, '-', label='result')
    plt.xlabel('t(sec)')
    plt.ylabel('position(m)')
    plt.legend(loc=0)
    plt.grid(True)

    if title:
        plt.title(title)

    plt.savefig(png_filename, format='png')
    plt.close()


def test_result_cost__is_float(param:np.ndarray, result_cost:float):
    assert isinstance(result_cost, float), f'param = {param}, result={result_cost}'


def test_result_cost__is_not_nan(param:np.ndarray, result_cost:float):
    assert math.isnan(result_cost) == False, f'param = {param}, result={result_cost} is Not a Number'


def test_result_cost__positive(param:np.ndarray, result_cost:float):
    assert result_cost >= 0.0, f'param = {param}, result={result_cost} is negative'


@pytest.mark.parametrize(
    "param_name, param_index, delta",
    [
        ('A', 0, lambda A: A + 10.0),
        ('zeta', 1, lambda zeta: 1 - zeta),
        ('w', 2, lambda w: w + 10.0),
        ('phi', 3, lambda phi: phi + 0.5 * np.pi),
        ('dc', 4, lambda w: w + 10.0)
    ]
)
def test_result_cost_sensitivity(
        param:np.ndarray, result_cost:float,
        t_x:Tuple[np.ndarray], t_x_dc:Tuple[np.ndarray], zeta:float, stdev:float,
        param_name:str, param_index:int, delta:Callable[[float], float]
    ):
    param2 = param.copy()
    param2[param_index] = delta(param2[param_index])
    result_cost_2 = wk06.wk06_cost(param2, *t_x)
    compare_plot(
        t_x[0], t_x[1], t_x_dc[1], wk06.wk06_curve(*param2, t_x[0]),
        f'param_{param_name}_{zeta:.4f}_{stdev:.4f}.png',
        f'result = {result_cost} result_A = {result_cost_2}'
    )
    assert result_cost_2 > result_cost, f', result={result_cost} when param = {param} is not smaller than result2={result_cost_2} when param = {param2}'


def test_result_curve(
        param:np.ndarray, result_curve:float,
        t_x:Tuple[np.ndarray],
        t_x_dc:Tuple[np.ndarray],
        zeta:float, stdev:float,
    ):
    assert isinstance(result_curve, (np.ndarray, list, tuple)), f'param = {param}, result={result_cost}'
    assert len(result_curve) == len(t_x_dc[0]), f'param = {param}, len(result)={len(result_curve)}, len(expected)={len(t_x_dc[0])}'

    png_filename = f'test_result_curve_{zeta:.4f}_{stdev:.4f}.png'
    compare_plot(t_x_dc[0], t_x[1], t_x_dc[1], result_curve, png_filename)

    nt.assert_allclose(result_curve, t_x_dc[1], err_msg=f"please check {png_filename} (possibly in the Artifact)")


if "__main__" == __name__:
    pytest.main([__file__])
