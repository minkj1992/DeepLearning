import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h))/(2*h)