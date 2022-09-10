# %%
import numpy as np 
from matplotlib import pyplot as plt 

np.random.seed(1)
pos_x = [0]
pos_y = [0]
pos_z = [0]
step_x=0
step_y=0
step_z =0
steps = 10000

for i in range(steps):
    #获取x的随机值
    step = 1 if np.random.randint(2) else -1
    step_x +=step
    pos_x .append(step_x)

    #获取y的随机值
    step = 1 if np.random.randint(2) else -1
    step_y +=step
    pos_y .append(step_y)

    #获取z的随机值
    step = 1 if np.random.randint(2) else -1
    step_z +=step
    pos_z .append(step_z)
#3d
plt.axes(projection = '3d')
plt.plot(pos_x,pos_y,pos_z)



# %%
