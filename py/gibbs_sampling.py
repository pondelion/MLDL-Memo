import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import os
%matplotlib inline


def f(x, y):
    # return np.exp(-(x**2 + y**2)/4)
    return np.sin(x)+np.cos(y)+2


def sample_x(y):
    x_range = np.arange(-10, 10, 0.02)
    f_vals = np.array([f(x, y) for x in x_range])
    accum_f_vals = np.array([f_vals[:i+1].sum() for i in range(len(f_vals))])
    accum_f_vals /= accum_f_vals[-1]
    rand = np.random.rand()
    return x_range[len(*np.where(accum_f_vals < rand))]


def sample_y(x):
    y_range = np.arange(-10, 10, 0.02)
    f_vals = np.array([f(x, y) for y in y_range])
    accum_f_vals = np.array([f_vals[:i+1].sum() for i in range(len(f_vals))])
    accum_f_vals /= accum_f_vals[-1]
    rand = np.random.rand()
    return y_range[len(*np.where(accum_f_vals < rand))]


xs = []
ys = []

x = np.random.rand()
y = np.random.rand()

xs.append(5)
ys.append(5)

for _ in range(20000):
    xs.append(sample_x(y=ys[-1]))
    ys.append(sample_y(x=xs[-1]))

plt.xlim(-10, 10)
plt.ylim(-10, 10)
# plt.scatter(xs, ys, s=3)
plt.hist2d(xs, ys, bins=40)
