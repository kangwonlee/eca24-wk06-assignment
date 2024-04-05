import wk06


def bisection(f, x_lower, x_upper, epsilon=1e-6,):
    while True:
        d = wk06.wk06(f, x_lower, x_upper, epsilon)

        if d['found']:
            break
        else:
            x_lower = d['x_lower']
            x_upper = d['x_upper']

    return d['x_lower']


def poly(x):
    return x * x - 20


def main():
    x = bisection(poly, 0, 100, 1e-6)
    print(f"poly({x}) = {poly(x)} is close to zero.")


if "__main__" == __name__:
    main()
