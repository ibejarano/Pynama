import sys
import petsc4py
from math import pi, sin, cos, exp, erf, sqrt
petsc4py.init(sys.argv)

from cases.base_problem import BaseProblem
import numpy as np
import yaml
from mpi4py import MPI
from petsc4py import PETSc
from viewer.paraviewer import Paraviewer
# from viewer.plotter import Plotter

class CustomFuncCase(BaseProblem):

    @staticmethod
    def taylorGreenVel_2D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(x_) * sin(y_) * expon, -sin(x_) * cos(y_) * expon]
        return [vel[0], vel[1]]

    @staticmethod
    def taylorGreenVort_2D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(y_) * expon
        return [vort]

    @staticmethod
    def taylorGreenVel_3D(coord, nu ,t=None):
        Lx= 1
        Ly= 1
        Lz = 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        vel = [cos(x_) * sin(y_) *sin(z_)*Lx* expon, sin(x_) * cos(y_) *sin(z_)*Ly *expon,-2*sin(x_)* sin(y_) * cos(z_) *Lz* expon]
        return vel

    @staticmethod
    def taylorGreenVort_2D_3D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vort = -2 * pi * (1.0 / Lx + 1.0 / Ly) * cos(x_) * cos(y_) * expon
        return [0,0,vort]
    
    @staticmethod
    def taylorGreenVel_2D_3D(coord, nu,t=None):
        Lx= 1
        Ly= 1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2))
        vel = [cos(x_) * sin(y_) * expon, -sin(x_) * cos(y_) * expon]
        return [vel[0], vel[1],0]

    @staticmethod
    def taylorGreenVort_3D(coord, nu ,t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        vort = [-2 * pi * (Ly / Lz + 2* Lz / Ly) * sin(x_) * cos(y_) *cos(z_)* expon,2 * pi * (Lx / Lz + 2* Lz / Lx) * cos(x_) * sin(y_) *cos(z_)* expon,2 * pi * (Ly / Lx - Lx / Ly) * cos(x_) * cos(y_) *sin(z_)* expon]
        return vort

    @staticmethod
    def taylorGreen3dConvective(coord, nu, t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        conv = [-2*(2*Lz/Ly+Ly/Lz)*( 2 * pi *expon)**2*sin(y_)*cos(y_)*sin(z_)*cos(z_), \
                 2*(2*Lz/Lx+Lx/Lz)*( 2 * pi *expon)**2*sin(x_)*cos(x_)*sin(z_)*cos(z_),\
                2*(2*Lx/Ly-2*Ly/Lx)*( 2 * pi *expon)**2*sin(y_)*cos(y_)*sin(x_)*cos(x_)]
        return conv

    @staticmethod
    def taylorGreen3dDiffusive(coord, nu, t=None):
        Lx= 1
        Ly= 1
        Lz=1
        Uref = 1
        x_ = 2 * pi * coord[0] / Lx
        y_ = 2 * pi * coord[1] / Ly
        z_ = 2 * pi * coord[2] / Lz
        expon = Uref * nu* exp(-4 * (pi**2) * nu * t * (1.0 / Lx ** 2 + 1.0 / Ly ** 2+ 1.0 / Lz ** 2))
        diff = [(2*pi)**3*expon*sin(x_)*cos(y_)*cos(z_)*(2*(Lz/(Lx*Lx*Ly)+Lz/(Ly*Ly*Ly)+Lz/(Lz*Lz*Ly))+Ly/(Lx*Lx*Lz)+Ly/(Ly*Ly*Lz)+Ly/(Lz*Lz*Lz)),\
            -(2*pi)**3*expon*cos(x_)*sin(y_)*cos(z_)*(2*(Lz/(Lx*Lx*Lx)+Lz/(Ly*Ly*Lx)+Lz/(Lz*Lz*Lx))+Lx/(Lx*Lx*Lz)+Lx/(Ly*Ly*Lz)+Lx/(Lz*Lz*Lz)),\
             (2*pi)**3*expon*cos(x_)*cos(y_)*sin(z_)*(Lx/(Lx*Lx*Ly)+Lx/(Ly*Ly*Ly)+Lx/(Lz*Lz*Ly)-Ly/(Lx*Lx*Lx)-Ly/(Ly*Ly*Lx)-Ly/(Lz*Lz*Lx))]
        return diff

    @staticmethod
    def senoidalVel_2D(coord, nu,t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        vel = [sin(x_), sin(y_)]
        return [vel[0], vel[1]]

    @staticmethod
    def senoidalVort_2D(coord, nu,t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        vort = Wref_x * pi *  cos(y_) - Wref_y * pi * cos(x_)
        return [vort]

    @staticmethod
    def senoidalConvective(coord, nu, t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        conv = ((Wref_y * pi)**2 - (Wref_x * pi )**2) * sin(x_) * sin(y_)
        return [conv]

    @staticmethod
    def senoidalDiffusive(coord, nu, t=None):
        Wref_x = 4
        Wref_y = 2
        x_ = Wref_y * pi * coord[1]
        y_ = Wref_x * pi * coord[0]
        tmp1 = -(Wref_x *pi)**3 * cos(y_) 
        tmp2 = (Wref_y *pi)**3 * cos(x_) 
        return [tmp1 + tmp2]

    @staticmethod
    def flatplateVel(coord, nu , t=None):
        U_ref = 1
        vx = U_ref * erf(coord[1]/ sqrt(4*nu*t))
        vy = 1
        return [vx, vy]

    @staticmethod
    def flatplateVort(coord, nu, t=None):
        tau = sqrt(4*nu*t)
        vort = (-2/(tau * sqrt(pi))) * exp(-(coord[1]/tau)**2)
        return [vort]

    @staticmethod
    def flatplateConvective(coord, nu, t=None):
        c = 1
        tau = sqrt(4*nu*t)
        alpha = 4 * c * coord[1] / ( sqrt(pi) * tau**3 )
        convective = alpha * exp( -(coord[1]/tau)**2 )
        return [convective]

    @staticmethod
    def flatplateDiffusive(coord, nu, t=None):
        tau = sqrt(4*nu*t)
        alpha = 4 / (sqrt(pi)* tau**3)
        beta = ( 1 - 2 * coord[1]**2 / tau**2 )
        diffusive = nu * alpha * beta * exp( -(coord[1]/tau)**2 )
        return [diffusive]