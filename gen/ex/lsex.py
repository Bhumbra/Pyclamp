
import numpy as np
from scipy.optimize import leastsq

def f(var,xs):
    return var[0]*np.exp(-var[1]*xs)+var[2]

def func(var, xs, ys):
    return f(var,xs) - ys

def dfunc(var,xs,ys):
    v = np.exp(-var[1]*xs)
    return [v,-var[0]*xs*v,np.ones(len(xs))]

xs = np.linspace(0,4,50)
ys = f([2.5,1.3,0.5],xs)
yn = ys + 0.2*np.random.normal(size=len(xs))
fit = leastsq(func,[10,10,10],args=(xs,yn),Dfun=dfunc,col_deriv=1)
