import numpy as np,matplotlib.pyplot as plt
import torch
import torch.nn as nn,torch.nn.functional as F

theta = torch.tensor([0.0],requires_grad=True)

f = lambda x: F.sigmoid((x-5)*2)

def gamma (proba,u):
    if 0.5-(proba+u)>0:
        return 0
    else: return 1

u = 0
learning_rate =0.1

F_=[]
G=[]
T=[]

for i in xrange(30000):
    print 'theta:%.2f, f(theta):%.2f, gamma:%.2f, u:%.2f'% (theta.item(),f(theta).item(),gamma(f(theta),u),u)
    l = (f(theta)-gamma(f(theta),u))**2
    l.backward()
    with torch.no_grad():
        theta-=theta.grad*learning_rate
        pass
    u = u + (f(theta) - gamma(f(theta),u))
    # print(theta.item(),f(theta).item(),gamma(f(theta),u),u.item())
plt.figure()
plt.plot(F_)
plt.plot(G)
plt.plot(T)
plt.show()

