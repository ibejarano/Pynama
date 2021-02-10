from math import exp, sin, cos, pi
import numpy as np

def alpha(nu, t):
    Uref = 1
    Lx = 1
    Ly = 1
    return Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))

def velocity(coords, alpha=1):
    vel = np.zeros(coords.shape)
    Lx= 1
    Ly= 1
    x_ = 2 * pi * coords[:,0] / Lx
    y_ = 2 * pi * coords[:,1] / Ly
    vel[:,0] = np.cos(x_) * np.sin(y_) * alpha
    vel[:,1] = -np.sin(x_) * np.cos(y_) * alpha
    return vel.flatten()

def velocity_test(coord, t, nu):
    Lx= 1
    Ly= 1
    Uref = 1
    x_ = 2 * pi * coord[0] / Lx
    y_ = 2 * pi * coord[1] / Ly
    expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
    vel = [cos(x_) * sin(y_) * expon, -sin(x_) * cos(y_) * expon]
    return vel

def vorticity(coord, alpha):
    Lx= 1
    Ly= 1
    x_ = 2 * pi * coord[:,0] / Lx
    y_ = 2 * pi * coord[:,1] / Ly
    vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * np.cos(x_) * np.cos(y_) * alpha
    return vort

def vorticity_test(coord, nu, t):
    Lx= 1
    Ly= 1
    Uref = 1
    x_ = 2 * pi * coord[0] / Lx
    y_ = 2 * pi * coord[1] / Ly
    expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
    vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(y_) * expon
    return [vort]

def convective(coord, alpha):
    pass

def diffusive(coord, alpha):
    pass