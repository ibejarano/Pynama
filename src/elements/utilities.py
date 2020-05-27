from collections import namedtuple
from itertools import combinations, permutations, product
import numpy as np
from math import pi
from numpy.linalg import eig

class GaussPoint2D(namedtuple('GaussPoint', ['r' ,'s','w'])):
    """ docstring of gausspoint """
    __slots__ = ()

class GaussPoint3D(namedtuple('GaussPoint',['r' ,'s','t','w'] )):
    """ docstring of gausspoint """
    __slots__ = ()

def generateGaussPoints2D(gps1D):
    gaussPoints = list()
    gpNamed = namedtuple('GaussPoint','r s w' )
    for gaussPointsCombination in product( gps1D , repeat=2):
        weight = 1
        for gaussPoint in gaussPointsCombination:
            weight *= gaussPoint
        gaussPoints.append(gpNamed(r=gaussPointsCombination[0], s=gaussPointsCombination[1], w=(weight)))
    return gaussPoints

def generateGaussPoints3D(gps1D):
    gaussPoints = list()
    gpNamed = namedtuple('GaussPoint','r s t w' )
    for gaussPointsCombination in product( gps1D , repeat=3):
        weight = 1
        for gaussPoint in gaussPointsCombination:
            weight *= gaussPoint
        gaussPoints.append(gpNamed(r=gaussPointsCombination[0], s=gaussPointsCombination[1], t=gaussPointsCombination[2], w=(weight)))
    return gaussPoints

def gaussPoints(N):
    """Calculate the properties of the Gauss points in 1D.

    :param N: Number of Gauss points in 1D.
    :type N: int
    :returns: Tuple with position of the points and their weight.
    :rtype: tuple

    """
    beta = 0.5 / np.sqrt(1.0 - (2.0 * np.arange(1, N)) ** -2)
    T = np.diag(beta, 1) + np.diag(beta, -1)
    [W, V] = eig(T)
    i = np.argsort(W)
    x = W[i]
    w = 2 * V[0, i] ** 2
    # Symmetrize Gauss points
    x = (x - x[::-1]) / 2
    w = (w + w[::-1]) / 2
    return (x, w)

def lobattoPoints(N):
    """Calculate the properties of the Gauss-Lobatto points in 1D.

    :param N: Number of Gauss-Lobatto points in 1D.
    :type N: int
    :returns: Tuple with position of the points and their weight.
    :rtype: tuple

    """
    # Use the Chebyshev-Gauss-Lobatto nodes as the first guess
    x = np.cos(np.linspace(0, pi, N))
    # The Legendre Vandermonde Matrix
    P = np.zeros((N, N))
    # Compute P_(N) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = 2
    while np.amax(np.abs(x - xold)) > (1e-15):
        xold = x
        P[:, 0] = 1
        P[:, 1] = x
        for k in range(2, N):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] -
                        (k - 1) * P[:, k - 2]) / k
        x = xold - (x * P[:, N - 1] - P[:, N - 2]) / (N * P[:, N - 1])

    w = 2.0 / ((N - 1) * N * np.square(P[:, N - 1]))
    x = (x[::-1] - x) / 2
    w = (w[::-1] + w) / 2
    return (x, w)
