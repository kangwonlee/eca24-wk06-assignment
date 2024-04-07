import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import scipy.optimize as so
import scipy.signal as ss


import wk06


def get_2nd_order_system(m_kg=1.0, k_Nm=1000.0, c_Nms=5.0,):
    mA = np.array([[0, 1.0], [-k_Nm/m_kg, -c_Nms/m_kg]])
    mB = np.array([[0], [1.0/m_kg]])
    mC = np.array([[1.0, 0.0]])
    mD = np.array([[0.0]])

    return ss.StateSpace(mA, mB, mC, mD)


def get_2nd_order_system_response(t0_sec=0.0, te_sec=2., sample_rate_Hz=10000):
    n_sample = int((te_sec - t0_sec) * sample_rate_Hz) + 1
    sys = get_2nd_order_system()
    t_sec = np.linspace(t0_sec, te_sec, n_sample)
    u_N = np.zeros_like(t_sec)
    t_sec, x_m, q = ss.lsim(sys, u_N, t_sec, X0=np.array([0.9, 0]))

    x_m += nr.normal(0.0, 0.1, t_sec.shape)
    x_m += 0.05

    return t_sec, x_m, q


def main():
    t_sec, x_m, q = get_2nd_order_system_response()
    # try to fit the response with the function wk06.wk06

    # possible upper and lower bounds of the parameters
    bounds = [
        (0, None),          # A
        (0, 0.5),           # zeta
        (0, None),          # w
        (-np.pi, np.pi),    # phi
        (None, None)        # offset
    ]

    result = so.minimize(
        wk06.wk06_cost,
        x0=np.array([0.8, 0.05, 40.0, 0.6, 0.1]),
        args=(t_sec, x_m),
        bounds=bounds
    )

    print(result)

    param = result.x

    x_m_final = wk06.wk06_curve(*param, t_sec)

    plt.plot(t_sec, x_m, label='measurement')
    plt.plot(t_sec, x_m_final, label='fitting result')
    plt.xlabel('t(sec)')
    plt.ylabel('x(m)')
    plt.legend(loc=0)
    plt.grid(True)
    plt.savefig('fitting_result.png')


if "__main__" == __name__:
    main()
