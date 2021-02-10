from math import exp, sin, cos, pi
import numpy as np 

def alpha(nu, t):
    Uref = 1
    Lx = 1
    Ly = 1
    Lz = 1
    return  Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))

def velocity(coord, alpha):
    vel = np.zeros(coord.shape)
    Lx= 1
    Ly= 1
    Lz = 1
    x_ = 2 * pi * coord[:,0] / Lx
    y_ = 2 * pi * coord[:,1] / Ly
    z_ = 2 * pi * coord[:,2] / Lz
    vel[:,0] = np.cos(x_) * np.sin(y_) *np.sin(z_)*Lx* alpha 
    vel[:,1] = np.sin(x_) * np.cos(y_) *np.sin(z_)*Ly *alpha 
    vel[:,2] = -2*np.sin(x_)* np.sin(y_) * np.cos(z_) *Lz* alpha 
    return vel.flatten()

def vorticity(coord, alpha):
    vort = np.zeros(coord.shape)
    Lx, Ly, Lz = (1, 1, 1)
    x_ = 2 * pi * coord[:,0] / Lx
    y_ = 2 * pi * coord[:,1] / Ly
    z_ = 2 * pi * coord[:,2] / Lz
    vort[:,0] = -2 * pi * (Ly / Lz + 2* Lz / Ly) * np.sin(x_) * np.cos(y_) *np.cos(z_)* alpha 
    vort[:,1] =  2 * pi * (Lx / Lz + 2* Lz / Lx) * np.cos(x_) * np.sin(y_) *np.cos(z_)* alpha 
    vort[:,2] =  2 * pi * (Ly / Lx - Lx / Ly) * np.cos(x_) * np.cos(y_) *np.sin(z_)* alpha 
    return vort.flatten()

def convective(coord, alpha):
    conv = np.zeros(coord.shape)
    Lx, Ly, Lz = (1, 1, 1)
    x_ = 2 * pi * coord[:,0] / Lx
    y_ = 2 * pi * coord[:,1] / Ly
    z_ = 2 * pi * coord[:,2] / Lz
    conv[:,0] = -2*(2*Lz/Ly+Ly/Lz)*( 2 * pi *alpha)**2*np.sin(y_)*np.cos(y_)*np.sin(z_)*np.cos(z_)
    conv[:,1] = 2*(2*Lz/Lx+Lx/Lz)*( 2 * pi *alpha)**2*np.sin(x_)*np.cos(x_)*np.sin(z_)*np.cos(z_)
    conv[:,2] = 2*(2*Lx/Ly-2*Ly/Lx)*( 2 * pi *alpha)**2*np.sin(y_)*np.cos(y_)*np.sin(x_)*np.cos(x_)
    return conv.flatten()

def diffusive(coord, alpha):
    difv = np.zeros(coord.shape)
    Lx, Ly, Lz = (1, 1, 1)
    x_ = 2 * pi * coord[:,0] / Lx
    y_ = 2 * pi * coord[:,1] / Ly
    z_ = 2 * pi * coord[:,2] / Lz
    difv[:,0] = (2*pi)**3*alpha*np.sin(x_)*np.cos(y_)*np.cos(z_)*(2*(Lz/(Lx*Lx*Ly)+Lz/(Ly*Ly*Ly)+Lz/(Lz*Lz*Ly))+Ly/(Lx*Lx*Lz)+Ly/(Ly*Ly*Lz)+Ly/(Lz*Lz*Lz))
    difv[:,1] = -(2*pi)**3*alpha*np.cos(x_)*np.sin(y_)*np.cos(z_)*(2*(Lz/(Lx*Lx*Lx)+Lz/(Ly*Ly*Lx)+Lz/(Lz*Lz*Lx))+Lx/(Lx*Lx*Lz)+Lx/(Ly*Ly*Lz)+Lx/(Lz*Lz*Lz))
    difv[:,2] = (2*pi)**3*alpha*np.cos(x_)*np.cos(y_)*np.sin(z_)*(Lx/(Lx*Lx*Ly)+Lx/(Ly*Ly*Ly)+Lx/(Lz*Lz*Ly)-Ly/(Lx*Lx*Lx)-Ly/(Ly*Ly*Lx)-Ly/(Lz*Lz*Lx))
    return difv.flatten()