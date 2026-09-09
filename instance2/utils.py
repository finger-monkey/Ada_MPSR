





import pandas as pd
import numpy as np

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

def Instance2_onedata(toolbox,x,y,z,t,u1):

    x0, y0,z0,t0 = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')

    expr_u1 = toolbox.compile(expr=u1)
    expr_u1 = expr_u1(x0, y0, z0,t0)
    expr_u1 = sympytorch.SymPyModule(expressions=[expr_u1])





    x = torch.from_numpy(x).clone().detach().requires_grad_(True)
    y = torch.from_numpy(y).clone().detach().requires_grad_(True)
    z= torch.from_numpy(z).clone().detach().requires_grad_(True)
    t = torch.from_numpy(t).clone().detach().requires_grad_(True)
    u1_expr = expr_u1(x=x,y=y,z=z,t=t)
    u1_expr = u1_expr.reshape(x.shape)
    return u1_expr.detach().numpy()

def Instance2_data(toolbox,x,y,z,t,u1,u2):

    x0, y0,z0,t0 = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')

    expr_u1 = toolbox.compile(expr=u1)
    expr_u1 = expr_u1(x0, y0, z0,t0)
    expr_u1 = sympytorch.SymPyModule(expressions=[expr_u1])

    expr_u2 = toolbox.compile(expr=u2)
    expr_u2 = expr_u2(x0, y0, z0,t0)
    expr_u2 = sympytorch.SymPyModule(expressions=[expr_u2])



    x = torch.from_numpy(x).clone().detach().requires_grad_(True)
    y = torch.from_numpy(y).clone().detach().requires_grad_(True)
    z= torch.from_numpy(z).clone().detach().requires_grad_(True)
    t = torch.from_numpy(t).clone().detach().requires_grad_(True)



    u1_expr = expr_u1(x=x,y=y,z=z,t=t)
    u1_expr = u1_expr.reshape(x.shape)
    u2_expr = expr_u2(x=x,y=y,z=z,t=t)
    u2_expr = u2_expr.reshape(x.shape)
    return u1_expr.detach().numpy(), u2_expr.detach().numpy()

def Instance2_PDEs(toolbox,x,y,z,t,u1,u2,match_path="comsol_instance1"):

    x0, y0,z0,t0 = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')

    expr_u1 = toolbox.compile(expr=u1)
    expr_u1 = expr_u1(x0, y0, z0,t0)
    expr_u1 = sympytorch.SymPyModule(expressions=[expr_u1])

    expr_u2 = toolbox.compile(expr=u2)
    expr_u2 = expr_u2(x0, y0, z0,t0)
    expr_u2 = sympytorch.SymPyModule(expressions=[expr_u2])



    x = torch.from_numpy(x).clone().detach().requires_grad_(True)
    y = torch.from_numpy(y).clone().detach().requires_grad_(True)
    z= torch.from_numpy(z).clone().detach().requires_grad_(True)
    t = torch.from_numpy(t).clone().detach().requires_grad_(True)



    u1_expr = expr_u1(x=x,y=y,z=z,t=t)
    u1_expr = u1_expr.reshape(x.shape)
    u2_expr = expr_u2(x=x,y=y,z=z,t=t)
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
        u1_y = torch.autograd.grad(u1_expr.sum(), y, create_graph=True)[0]
        try:
            u1_yy = torch.autograd.grad(u1_y.sum(), y, create_graph=True)[0]
        except:
            u1_yy = torch.zeros(y.size(), dtype=x.dtype)
    except:
        u1_y = torch.zeros(x.size(), dtype=x.dtype)
        u1_yy = torch.zeros(x.size(), dtype=x.dtype)


    try:
        u1_z = torch.autograd.grad(u1_expr.sum(), z, create_graph=True)[0]
        try:
            u1_zz = torch.autograd.grad(u1_z.sum(), z, create_graph=True)[0]
        except:
            u1_zz = torch.zeros(z.size(), dtype=x.dtype)
    except:
        u1_z = torch.zeros(x.size(), dtype=x.dtype)
        u1_zz = torch.zeros(x.size(), dtype=x.dtype)





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
        u2_y = torch.autograd.grad(u2_expr.sum(), y, create_graph=True)[0]
        try:
            u2_yy = torch.autograd.grad(u2_y.sum(), y, create_graph=True)[0]
        except:
            u2_yy = torch.zeros(y.size(), dtype=x.dtype)
    except:
        u2_y = torch.zeros(x.size(), dtype=x.dtype)
        u2_yy = torch.zeros(x.size(), dtype=x.dtype)


    try:
        u2_z = torch.autograd.grad(u2_expr.sum(), z, create_graph=True)[0]
        try:
            u2_zz = torch.autograd.grad(u2_z.sum(), z, create_graph=True)[0]
        except:
            u2_zz = torch.zeros(z.size(), dtype=x.dtype)
    except:
        u2_z = torch.zeros(x.size(), dtype=x.dtype)
        u2_zz = torch.zeros(x.size(), dtype=x.dtype)


    try:
        u2_t = torch.autograd.grad(u2_expr.sum(), t, create_graph=True)[0]
    except:
        u2_t = torch.zeros(t.size(), dtype=t.dtype)


    if match_path == "comsol_instance1":

        error1 = 200*(-u1_xx-u1_yy-u1_zz)
        error2 = 1e5 * u2_t - 1*(u2_xx + u2_yy + u2_zz) - 200 * (u2_x ** 2 + u2_y ** 2 + u2_z ** 2)
    elif match_path == "comsol_instance2":
        error1 = 100 * (-u1_xx - u1_yy - u1_zz)
        error2 = 1e5 * u2_t - 1 * (u2_xx + u2_yy + u2_zz) - 100 * (u2_x ** 2 + u2_y ** 2 + u2_z ** 2)
    elif match_path == "comsol_instance3":
        error1 = 100 * (-u1_xx - u1_yy - u1_zz)
        error2 = 1e2 * u2_t - 1 * (u2_xx + u2_yy + u2_zz) - 100 * (u2_x ** 2 + u2_y ** 2 + u2_z ** 2)
    elif match_path == "comsol_instance4":
        error1 = 100 * (-u1_xx - u1_yy - u1_zz)
        error2 = 1e1 * u2_t - 1 * (u2_xx + u2_yy + u2_zz) - 100 * (u2_x ** 2 + u2_y ** 2 + u2_z ** 2)
    elif match_path == "comsol_instance5":
        error1 = 100 * (-u1_xx - u1_yy - u1_zz)
        error2 = 1e1 * u2_t - 2 * (u2_xx + u2_yy + u2_zz) - 100 * (u2_x ** 2 + u2_y ** 2 + u2_z ** 2)

    mse1 = F.mse_loss(error1, torch.zeros_like(error1))
    mse2 = F.mse_loss(error2, torch.zeros_like(error2))


    return [(mse1.item()+mse2.item())/2.0]

def read_data(path):
    """

    df = pd.read_csv(path+".csv")


    x = df["X"].values
    y = df["Y"].values
    z = df["Z"].values
    t = df["t"].values
    V = df["V"].values
    T = df["T"].values

    return x,y,z,t,V,T

def main(path):



    x, y, z, t, u1, u2 = read_data(path)


    coords = np.vstack([x, y, z, t])





    sympy_expressions_u1 = []
    sympy_expressions_u2 = []
    index = [i for i in range(len(coords[0]))]
    for i in range(5):
        sample = np.random.choice(index, size=100, replace=False)
        X = coords[:, sample[:]]
        yy = u1[sample[:]]
        expression, logs = physo.SR(X, yy,

                                    X_names=["x", "y", "z", "t"],

                                    y_name="u1",
                                    free_consts_names=["m", "n", "q", "r", "s"],
                                    free_consts_units=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],

                                    run_config=physo.config.config0.config0,
                                    op_names=["mul", "add", "sub", "sin"]
                                    )


        sympy_expressions_u1 = sympy_expressions_u1 + physo.read_pareto_csv("SR_curves_pareto.csv")

        yy = u2[sample[:]]
        expression, logs = physo.SR(X, yy,

                                    X_names=["x", "y", "z", "t"],

                                    y_name="u1",
                                    free_consts_names=["m", "n", "q", "r", "s"],
                                    free_consts_units=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],

                                    run_config=physo.config.config0.config0,
                                    op_names=["mul", "add", "sub", "sin"]
                                    )


        sympy_expressions_u2 = sympy_expressions_u2 + physo.read_pareto_csv("SR_curves_pareto.csv")



        with open(path + 'sympy_expressions_u1.pkl', 'wb') as f:
            pickle.dump(sympy_expressions_u1, f)
        with open(path + 'sympy_expressions_u2.pkl', 'wb') as f:
            pickle.dump(sympy_expressions_u2, f)


if __name__ == '__main__':
    paths = ["comsol_instance1","comsol_instance2","comsol_instance3","comsol_instance4","comsol_instance5"]

    main("comsol_instance1")




