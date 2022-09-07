import numpy as np


def schaffer_f6(point: np.ndarray) -> np.ndarray:
    """Schaffer's f6 function

    minimum on f(0,0) = 0

    search space: -100 <= x,y <= 100
    """
    sqr_sum = (point**2).sum(axis=1)
    num = np.sin(np.sqrt(sqr_sum)) ** 2 - 0.5
    den = (1 + 0.001 * sqr_sum) ** 2
    return 0.5 + (num / den)


def ackley(point: np.ndarray) -> np.ndarray:
    """Ackley function

    minimum on f(0,0) = 0

    search space: -5 <= x,y <= 5
    """
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (point**2).sum(axis=1)))
        - np.exp(0.5 * (np.cos(2 * np.pi * point).sum(axis=1)))
        + np.e
        + 20
    )


def himmelblau(point: np.ndarray) -> np.ndarray:
    """Himmelblau function

    4 minima
        * f(3.0, 2.0) = 0
        * f(-2.805118, 3.131312) = 0
        * f(-3.779310, -3.283186) = 0
        * f(3.584428, -1.848126) = 0

    search space: -5 <= x,y <= 5
    """
    sqr_point = point**2
    return (sqr_point[:, 0] + point[:, 1] - 11) ** 2 + (
        point[:, 0] + sqr_point[:, 1] - 7
    ) ** 2
