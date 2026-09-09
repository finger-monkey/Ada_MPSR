
















from scipy.optimize import minimize
from deap import gp
import dis
import inspect
import random

def loss1d(params,ind,X1,y,pset):
    i = 0

    for node in ind:


        if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):

            node.value = params[i]
            i+=1
    ind.expr = gp.compile(ind, pset)


    y_pred = [ind.expr(X1[0][i],X1[1][i]) for i in range(len(X1[0]))]
    return sum([(a - b)**2 for a, b in zip(y, y_pred)])/len(y)


def loss2d(params,ind,X1,y,pset):
    i = 0

    for node in ind:


        if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):

            node.value = params[i]
            i+=1
    ind.expr = gp.compile(ind, pset)


    y_pred = [ind.expr(X1[0][i],X1[1][i],X1[2][i]) for i in range(len(X1[0]))]
    return sum([(a - b)**2 for a, b in zip(y, y_pred)])/len(y)



def loss3d(params,ind,X1,y,pset):
    i = 0

    for node in ind:


        if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):

            node.value = params[i]
            i+=1
    ind.expr = gp.compile(ind, pset)


    y_pred = [ind.expr(X1[0][i],X1[1][i],X1[2][i],X1[3][i]) for i in range(len(X1[0]))]
    return sum([(a - b)**2 for a, b in zip(y, y_pred)])/len(y)

def local_optimize(individual, X, y,pset):
    """


    params = [1 for node in individual if (isinstance(node, gp.Terminal) and not isinstance(node.value, str))]

    if len(params) == 0:
        return





    bounds = [(-4, 4) for i in range(len(params))]
    if len(X) == 2:
        minimize(loss1d, params, args=(individual, X, y, pset), method='L-BFGS-B',options={'maxiter': 4, 'gtol': 1e-3, 'disp': False})
    elif len(X) == 3:
        minimize(loss2d, params, args=(individual, X, y, pset), method='L-BFGS-B',options={'maxiter': 4, 'gtol': 1e-3, 'disp': False})
    elif len(X) == 4:
        minimize(loss3d, params, args=(individual, X, y, pset), method='L-BFGS-B',options={'maxiter': 4, 'gtol': 1e-3, 'disp': False})

    return



