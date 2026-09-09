








import numpy as np
import matplotlib.pyplot as plt
import torch

import physo
import sympytorch
from sympy import *
import sympy
import torch.nn.functional as F
import math

import pickle
import torch
from torch.nn import L1Loss

def Instance1_onedata(toolbox,x,t,u1):
    x0, t0 = Symbol('x'), Symbol('t')

    expr_u1 = toolbox.compile(expr=u1)
    expr_u1 = expr_u1(x0, t0)
    if isinstance(expr_u1,float):
        u1_expr= torch.full((len(x),),expr_u1)
    else:
        expr_u1 = sympytorch.SymPyModule(expressions=[expr_u1])


        x = torch.from_numpy(x).clone().detach().requires_grad_(True)
        t = torch.from_numpy(t).clone().detach().requires_grad_(True)

        u1_expr = expr_u1(x=x, t=t)
        u1_expr = u1_expr.reshape(x.shape)

    return u1_expr.detach().numpy()

def Instance1_data(toolbox,x,t,u1,u2):
    x0, t0 = Symbol('x'), Symbol('t')

    expr_u1 = toolbox.compile(expr=u1)
    expr_u1 = expr_u1(x0, t0)
    expr_u1 = sympytorch.SymPyModule(expressions=[expr_u1])

    expr_u2 = toolbox.compile(expr=u2)
    expr_u2 = expr_u2(x0, t0)
    expr_u2 = sympytorch.SymPyModule(expressions=[expr_u2])


    x = torch.from_numpy(x).clone().detach().requires_grad_(True)
    t = torch.from_numpy(t).clone().detach().requires_grad_(True)

    u1_expr = expr_u1(x=x, t=t)
    u1_expr = u1_expr.reshape(x.shape)
    u2_expr = expr_u2(x=x, t=t)
    u2_expr = u2_expr.reshape(x.shape)
    return u1_expr.detach().numpy() , u2_expr.detach().numpy()

def Instance1_PDEs(toolbox,x,t,u1,u2,match_path="data"):
    x0, t0 = Symbol('x'), Symbol('t')

    expr_u1 = toolbox.compile(expr=u1)
    expr_u1 = expr_u1(x0, t0)
    expr_u1 = sympytorch.SymPyModule(expressions=[expr_u1])

    expr_u2 = toolbox.compile(expr=u2)
    expr_u2 = expr_u2(x0, t0)
    expr_u2 = sympytorch.SymPyModule(expressions=[expr_u2])



    x = torch.from_numpy(x).clone().detach().requires_grad_(True)
    t = torch.from_numpy(t).clone().detach().requires_grad_(True)



    u1_expr = expr_u1(x=x,t=t)
    u1_expr = u1_expr.reshape(x.shape)
    u2_expr = expr_u2(x=x, t=t)
    u2_expr = u2_expr.reshape(x.shape)




    try:
        u1_x = torch.autograd.grad(u1_expr.sum(), x, create_graph=True)[0]
        try:
            u1_xx = torch.autograd.grad(u1_x.sum(), x, create_graph=True)[0]
        except:
            u1_xx = torch.zeros(x.size(), dtype=x.dtype)
    except:
        u1_x = torch.zeros(x.size(), dtype=x.dtype)
        u1_xx = torch.zeros(x.size(), dtype=x.dtype)




    try:
        u1_t = torch.autograd.grad(u1_expr.sum(), t, create_graph=True)[0]
    except:
        u1_t = torch.zeros(t.size(), dtype=t.dtype)




    try:
        u2_x = torch.autograd.grad(u2_expr.sum(), x, create_graph=True)[0]
        try:
            u2_xx = torch.autograd.grad(u2_x.sum(), x, create_graph=True)[0]
        except:
            u2_xx = torch.zeros(x.size(), dtype=x.dtype)
    except:
        u2_x = torch.zeros(x.size(), dtype=x.dtype)
        u2_xx = torch.zeros(x.size(), dtype=x.dtype)


    try:
        u2_t = torch.autograd.grad(u2_expr.sum(), t, create_graph=True)[0]
    except:
        u2_t = torch.zeros(t.size(), dtype=t.dtype)


    if match_path == "data":


        F_y = torch.exp(5.73 * (u1_expr - u2_expr)) - torch.exp(-11.46 * (u1_expr - u2_expr))
        error1 = u1_t-0.024*u1_xx+F_y
        error2 = u2_t-0.17*u2_xx-F_y
    elif match_path == "data_instance_2":
        F_y = torch.exp(5.73 * (u1_expr - u2_expr)) - torch.exp(-11.46 * (u1_expr - u2_expr))
        error1 = u1_t - 0.24 * u1_xx + F_y
        error2 = u2_t - 0.17 * u2_xx - F_y
    elif match_path == "data_instance_3":
        F_y = u1_expr - u2_expr
        error1 = u1_t - 1.7 * u1_xx + F_y
        error2 = u2_t - 0.17 * u2_xx - F_y

    mse1 = F.mse_loss(error1, torch.zeros_like(error1))
    mse2 = F.mse_loss(error2, torch.zeros_like(error2))


    return [(mse1.item()+mse2.item())/2.0]


def AD_test():
    """
    x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
    t = torch.tensor([1.0,2.0,3.0],requires_grad=True)
    u = x*x+t
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]

    print(u_xx,u_t)
    pass



def read_data(path):
    """
    import scipy.io as sio
    data = sio.loadmat(path+".mat")
    x = data['x'][0]
    t = data['t'][0]
    u1 = data['u1']
    u2 = data['u2']
    return x,t,u1,u2

def main(path):


    x, t, u1, u2 = read_data(path)


    X, T = np.meshgrid(x, t)

    coords = np.vstack([X.ravel(), T.ravel()])




    sympy_expressions_u1 = []
    sympy_expressions_u2 = []
    index = [i for i in range(len(coords[0]))]
    for i in range(5):
        sample = np.random.choice(index, size=100, replace=False)
        X = coords[:, sample[:]]
        y = u1.reshape(-1)[sample[:]]
        expression, logs = physo.SR(X, y,

                                    X_names=["x", "t"],

                                    y_name="u1",
                                    free_consts_names=["m", "n", "q"],
                                    free_consts_units=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],

                                    run_config=physo.config.config0.config0,
                                    op_names=["mul", "add", "sub", "sin"]
                                    )


        sympy_expressions_u1 = sympy_expressions_u1 + physo.read_pareto_csv("SR_curves_pareto.csv")

        y = u2.reshape(-1)[sample[:]]
        expression, logs = physo.SR(X, y,

                                    X_names=["x", "t"],

                                    y_name="u2",
                                    free_consts_names=["m", "n", "q"],
                                    free_consts_units=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],

                                    run_config=physo.config.config0.config0,
                                    op_names=["mul", "add", "sub", "sin"]
                                    )


        sympy_expressions_u2 = sympy_expressions_u2 + physo.read_pareto_csv("SR_curves_pareto.csv")


        with open(path+'sympy_expressions_u1.pkl', 'wb') as f:
            pickle.dump(sympy_expressions_u1, f)

        with open(path+'sympy_expressions_u2.pkl', 'wb') as f:
            pickle.dump(sympy_expressions_u2, f)


if __name__ == '__main__':
    paths = ["data","data_instance_2","data_instance_3"]
    main("data_instance_3")

