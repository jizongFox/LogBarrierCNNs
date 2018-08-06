# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

'''
this is to demonstrate the ADMM for image segmentations without interaction with each other
supposing you have a neural network give the possibility of being foreground or background.
if there's only one pixel, or again, no interactions between pixels, the graphcut would give you the result based on 0.5

Two objectives here in this script. 
1. see the stablity of only one pixels
2. verify that for an image without interactions between pixels, meaning no boundary term.
'''


# for one pixel case
def get_gamma(proba, u):
    if abs(1 - proba + u) > abs(0 - proba + u):
        return 0
    else:
        return 1


def method1():
    network = lambda theta: F.sigmoid(theta)

    theta = torch.randn((1, 1), requires_grad=True)
    probability = network(theta * 0.01)
    u = 0

    p = 1.0
    learning_rate = 0.01

    p_list = []
    gamma_list = []
    u_list = []

    for i in range(1000):
        # nupdate gamma
        gamma = get_gamma(probability.data.numpy(), u)
        ## update theta:
        for i in range(5):
            # l = p / 2 * (torch.tensor(gamma).float() - probability + torch.tensor(u).float()).norm(2) ** 2
            l = u*(probability-torch.tensor(gamma).float()) + p/2* (probability-torch.tensor(gamma).float())**2
            l.backward()
            with torch.no_grad():
                theta -= theta.grad * learning_rate
                theta.grad.zero_()
            probability = network(theta)

        # u = u +(gamma - (probability.item()>0.5)*1.0)

        u = u + (probability.item()-gamma )

        print(probability.item(), gamma, u)
        p_list.append(probability.detach().item())
        gamma_list.append(gamma)
        u_list.append(u)

    plt.figure()
    plt.plot(p_list, label='probability')
    plt.plot(gamma_list, label='gamma')
    plt.figure()
    plt.plot(u_list, label='u')
    plt.legend()
    plt.show()


## it turns out that if f(theta) == gamma and gamma is no differential and no convex function, the ADMM gets stuck
## if you modify the constraint such as binarized (f) == gamma,that gives you a suprised.


'''
Another method to use the inequality constraint is that:
probabilty should be :
gamma - 0.5 + eps << probability << gamma + 0.5 - eps with eps being small
the constrain problem becomes:
loss = R(gamma) s.t.
prob << gamma +0.5 - eps
and 
-prob << -gamma + 0.5 - eps
meaning that prob == gamma +0.5 - eps - y1^2
and prob == gamma - 0.5 + eps + y2^2
method of multiplier:
L = R(gamma) + || prob - gamma - 0.5 + eps + y1^2 + u1||^2 + || prob - gamma + 0.5 - eps -y2^2 + u2||^2
update theta: 
l_theta =  || prob - gamma - 0.5 + eps + y1^2 +u1||^2 + || prob - gamma + 0.5 - eps -y2^2 + u2||^2
update gamma: 
just compare the proposals of gamma =0 or gamma =1 for || prob - gamma - 0.5 + eps + y1^2 + u1||^2 + || prob - gamma + 0.5 - eps -y2^2 +u2||^2
update y1^2:

y1^2 = max(-(prob - gamma - 0.5 + eps+ u1),0)
y2^2 = max(prob - gamma + 0.5 - eps +u2,0)

update multipilers

u1 =u1 +  prob - gamma - 0.5 + eps + y1^2 
u2 = u2 + prob - gamma + 0.5 - eps -y2^2 
'''


def get_gamma_2(prob_, eps_, y1_2_, y2_2_, u1_, u2_):
    gamma_1 = np.abs(prob_ - 1 - 0.5 + eps_ + y1_2_ + u1_) ** 2 + np.abs(prob_ - 1 + 0.5 - eps_ - y2_2_ + u2_) ** 2
    gamma_0 = np.abs(prob_ - 0 - 0.5 + eps_ + y1_2_ + u1_) ** 2 + np.abs(prob_ - 0 + 0.5 - eps_ - y2_2_ + u2_) ** 2
    if gamma_1 > gamma_0:
        return 0
    else:
        return 1

def inequality_method():
    learning_rate = 0.1
    theta = torch.randn((1,1),requires_grad=True)
    network = lambda x: F.sigmoid(theta)
    prob = network(theta)
    gamma = 0
    eps = 0.25
    y1_2= 0
    y2_2 = 0
    u1=0
    u2=0

    P=[]
    G=[]

    for i in range(100):
        # update theta
        # l_theta = | | prob - gamma - 0.5 + eps + y1 ^ 2 + u1 | | ^ 2 + | | prob - gamma + 0.5 - eps - y2 ^ 2 + u2 | | ^ 2
        l_theta = torch.abs(prob - torch.tensor(gamma + 0.5 - eps - y1_2 - u1).float()) ** 2 + torch.abs(prob +torch.tensor(- gamma + 0.5 - eps - y2_2 + u2).float()) ** 2
        l_theta.backward()
        with torch.no_grad():
            theta-= theta.grad * learning_rate
            theta.grad.zero_()
        prob = network(theta)
        P.append(prob.item())

        # update gamma
        gamma = get_gamma_2(prob.data.numpy(),eps,y1_2,y2_2,u1,u2)
        G.append(gamma)
        # update  y1 ^ 2:
        y1_2 = max(-(prob.data.numpy() - gamma - 0.5 + eps + u1), 0)
        y2_2 = max(prob.data.numpy() - gamma + 0.5 - eps + u2, 0)

        # update  multipilers

        u1 = u1 + prob.data.numpy() - gamma - 0.5 + eps + y1_2
        u2 = u2 + prob.data.numpy() - gamma + 0.5 - eps - y2_2

        print(prob.item(),gamma,u1, u2, y1_2, y2_2)

    plt.figure()
    plt.plot(P,label='proba')
    plt.plot(G,label='gamma')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    method1()
    # inequality_method()