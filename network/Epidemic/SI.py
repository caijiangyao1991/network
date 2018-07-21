# coding=utf-8
"""https://www.cnblogs.com/scikit-learn/p/6937326.html"""
import scipy.integrate as spi
import numpy as np
import pylab as pl

"""the likelihood that the disease will be transmitted from an infected to a susceptible
 9 individual in a unit time is β"""
beta = 1.4247
"""gamma is the recovery rate and in SI model, gamma equals zero"""
gamma= 0.14286
"""#I0 is the initial fraction of infected individuals"""
I0 = 1e-6
"""ND is the total time step"""
ND=70
TS=1.0

S0=1-1e-6 #易感染个体比例
I0=1e-6 #感染者比例

INPUT = (S0, I0, 0.0)

def diff_eqs(INP, t):
    '''The main set of equations SIR'''
    Y = np.zeros((3))
    V = INP
    Y[0] = -beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1]
    return Y

lammda=0.02
def diff_eqs1(INP,t):
    '''''The main set of equations SIRS'''
    Y=np.zeros((3))
    V = INP
    Y[0] = lammda * V[2] - beta * V[0] * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    Y[2] = gamma * V[1] - lammda * V[2]
    return Y   # For odeint

t_start = 0.0
t_end = ND
t_inc = TS

t_range = np.arange(t_start,t_end+t_inc,t_inc)
"""RES is the result of fraction of susceptibles and infectious individuals at each time step respectively"""
RES = spi.odeint(diff_eqs, INPUT, t_range)
print(RES)
pl.plot(RES[:,0], '-bs', label='Susceptibles')
pl.plot(RES[:,1], '-ro', label='Infectious')
pl.legend(loc=0)
pl.title('SI epidemic without births or deaths')
pl.xlabel('Time')
pl.ylabel('Susceptibles and Infectious')
pl.savefig('2.5-SI-high.png', dpi=900) # This does increase the resolution.
pl.show()





