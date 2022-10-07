# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:14:59 2022

@author: Chaofeng Wang
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"for newton"
def mhumps(x):
    return abs(-1/((x-0.3)**2+0.01)+1/((x-0.9)**2+0.04)-6)


"for GA"
def achley(x,y):
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * 
  np.pi * x)+np.cos(2 * np.pi * y))) + np.exp(1) + 20
                         
def rastrigin(x,y):
    return 20+ x**2-10*np.cos(2*np.pi*x)+y**2-10*np.cos(2*np.pi*y)


if __name__=='__main__':
    
    # plot mhumps
    x = np.arange(-10, 10, 0.01)
    y = mhumps(x)
    plt.plot(x,y)
    plt.show()

    #plot achley function
    r_min, r_max = -32.768, 32.768
    xaxis = np.arange(r_min, r_max, 2.0)
    yaxis = np.arange(r_min, r_max, 2.0)
    x, y = np.meshgrid(xaxis, yaxis)
    results = achley(x, y)
    figure = plt.figure()
    axis = figure.gca( projection='3d')
    axis.plot_surface(x, y, results, cmap='jet', shade= "false")
    plt.show()
    plt.contour(x,y,results)
    plt.show()


    #plot rastrigin function
    r_min, r_max = -5, 5
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)
    x, y = np.meshgrid(xaxis, yaxis)
    results1 = rastrigin(x, y)
    figure = plt.figure()
    axis = figure.gca( projection='3d')
    axis.plot_surface(x, y, results1, cmap='jet', shade= "false")
    plt.show()
    plt.contour(x,y,results1)
    plt.show()