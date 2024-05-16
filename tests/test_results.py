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

import cProfile
import io
import math
import pathlib
import pstats
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


import main


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
    return main.wk06_cost(param, *t_x)


@pytest.fixture
def result_curve(param:np.ndarray, t_x:Tuple[np.ndarray]) -> float:
    return main.wk06_curve(*param, t_x[0])


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


def test_cost_function_returns_float_value(param:np.ndarray, result_cost:float):
    assert isinstance(result_cost, float), (
        f'The cost function returned a non-float : {result_cost:g} param = {param}\n'
        f"비용 함수 결과 {result_cost:g} 가 float가 아님 : 입력매개변수 = {param}"
    )


def test_cost_function_returns_a_valid_number(param:np.ndarray, result_cost:float):
    assert math.isnan(result_cost) == False, (
        f'The cost function returned an invalid number : {result_cost:g} param = {param}\n'
        f"비용 함수 결과 {result_cost:g} 가 유효하지 않은 숫자임 : 입력매개변수 = {param}\n"
    )


def test_cost_function_returns_a_non_negative_number(param:np.ndarray, result_cost:float):
    assert result_cost >= 0.0, (
        f'The function returned a negative value : {result_cost:g} param = {param}.\n'
        f'비용 함수 결과 {result_cost:g} 가 음수임 : 입력매개변수 = {param}.'
    )


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
    result_cost_2 = main.wk06_cost(param2, *t_x)
    compare_plot(
        t_x[0], t_x[1], t_x_dc[1], main.wk06_curve(*param2, t_x[0]),
        f'param_{param_name}_{zeta:.4f}_{stdev:.4f}.png',
        f'result = {result_cost:g} result_A = {result_cost_2}'
    )
    assert result_cost_2 > result_cost, (
        f'result={result_cost:g} when param = {param} is not smaller than result2={result_cost_2:g} when param = {param2}'
        f'매개변수가 {param2} 인 경우의 비용함수 {result_cost_2:g} 에 비해 {param} 인 경우의 비용 함수 {result_cost:g} 가 더 작아져야 함'
    )


def test_result_curve_type(
        param:np.ndarray, result_curve:float,
    ):
    assert isinstance(result_curve, (np.ndarray, list, tuple)), (
        f'The curve function returned a {type(result_curve)}, '
        'not one of (np.ndarray, list, tuple) : '
        f'param = {param} result = {result_curve} \n'
        f'곡선 계산 함수의 반환값의 형이 (np.ndarray, list, tuple) 중 하나가 아님: '
        '입력매개변수 = {param}, 결과 = {result_curve}'
    )


def test_result_curve_dimension(
        param:np.ndarray, result_curve:float,
        t_x_dc:Tuple[np.ndarray],
    ):
    assert len(result_curve) == len(t_x_dc[0]), (
        f'The curve function returned {len(result_curve)} elements but expected {len(t_x_dc[0])} instead. param = {param}\n'
        f'곡선 계산 함수의 반환값에 요소 {len(result_curve)} 개가 포함되어 있었으나 {len(t_x_dc[0])} 개로 예상되었었음. 입력 매개변수 = {param}'
    )


def test_result_curve__values(
        param:np.ndarray, result_curve:float,
        t_x:Tuple[np.ndarray],
        t_x_dc:Tuple[np.ndarray],
        zeta:float, stdev:float,
    ):
    png_filename = f'test_result_curve_{zeta:.4f}_{stdev:.4f}.png'
    compare_plot(t_x_dc[0], t_x[1], t_x_dc[1], result_curve, png_filename)

    message = (
        f"The curve function returned results different from the expected. Please check {png_filename} (possibly in the Artifact) param={param}\n"
        f"곡선 계산 함수의 반환값이 예상과 다름. {png_filename}을 확인 바람 (아마도 Artifact에 있음) 입력매개변수 = {param}"
    )
    nt.assert_allclose(result_curve, t_x_dc[1], err_msg=message)


def run_cProfile():
    param = np.array([1.0, 0.1, 1.0, 0.0, 0.0])
    t_x = np.linspace(0, 10, 1001), np.sin(np.linspace(0, 10, 1001))

    main.wk06_cost(param, *t_x)
    main.wk06_curve(*param, t_x[0])


@pytest.mark.parametrize("solver_name", ['odeint', 'ode', 'solve_ivp'])
def test_no_ode_solver_used(solver_name):
    with cProfile.Profile() as pr:
        run_cProfile()

    output = io.StringIO()
    ps = pstats.Stats(pr, stream=output).sort_stats('tottime')

    for k in ps.stats:
        function_path = pathlib.Path(k[0]).resolve()
        if ('scipy' in function_path.parts) and ('integrate' in function_path.parts):
            assert k[-1] != solver_name, f'Probably this assignment would not need {k[-1]}()'


if "__main__" == __name__:
    pytest.main([__file__])
