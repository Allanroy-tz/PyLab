# %%
import numpy as np
import matplotlib.pyplot as plt
x1=0.000001
def mhumps(x):
    return abs(-1/((x-0.3)**2+0.01)+1/((x-0.9)**2+0.04)-6)
def newton(func,x0,k):
    _k = 0
    _xk = 0
    _xi = x0
    for i in range(k):
        q = GetDer(func,_xi)
        w = GetDoubelDer(func,_xi)
        _xk = _xi - q/w
        _xi=_xk
        _k+=1
    out =func(_xi)
    return _xi,out
#一阶导
def GetDer(func,x0):
    out = (func(x0+x1)-func(x0))/x1
    return out
#二阶导
def GetDoubelDer(func,x0):
    out = (GetDer(func,x0+x1)-GetDer(func,x0))/x1
    return out
if __name__=='__main__':
    
    # plot mhumps
    x = np.arange(-10, 10, 0.01)
    y = mhumps(x)
    plt.plot(x,y)
    plt.show()
    print(newton(mhumps,0,10))



