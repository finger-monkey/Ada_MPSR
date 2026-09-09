





from T_Baseline import *
import pandas as pd
import scipy.io as sio
import numpy as np
def main(path,method="T-Baseline"):



    x, t, u1, u2 = read_data(path)


    X, T = np.meshgrid(x, t)

    coords = np.vstack([X.ravel(), T.ravel()])
    X_train = coords
    y_train = [u1.reshape(-1),u2.reshape(-1)]

    toolbox, pset = define_gp()

    import pickle
    import os
    pkl_path_u1 = 'sympy_expressions_u1.pkl' if path in ['', '.'] else path + '_sympy_expressions_u1.pkl'
    pkl_path_u2 = 'sympy_expressions_u2.pkl' if path in ['', '.'] else path + '_sympy_expressions_u2.pkl'

    if not os.path.exists(pkl_path_u1):
        pkl_path_u1 = 'sympy_expressions_u1.pkl'
    if not os.path.exists(pkl_path_u2):
        pkl_path_u2 = 'sympy_expressions_u2.pkl'
    with open(pkl_path_u1, 'rb') as f:
        u1_exp = pickle.load(f)
    with open(pkl_path_u2, 'rb') as f:
        u2_exp  = pickle.load(f)



    if method == "T-Baseline":











        ind1 = str2ind("log(x + 1.0)", toolbox)
        ind2 = str2ind("0.76486668516552*sin(x)", toolbox)





        u1_data,u2_data = Instance1_data(toolbox, X_train[0], X_train[1], ind1, ind2)
        mae1 = np.mean(np.abs(u1_data - y_train[0]))
        var1 = np.var(u1_data - y_train[0])
        print(mae1, var1)
        mae2 = np.mean(np.abs(u2_data - y_train[1]))
        var2 = np.var(u2_data - y_train[1])
        print(mae2, var2)
        error = Instance1_PDEs(toolbox,X_train[0], X_train[1], ind1, ind2,match_path=path)
        print(error)


        my_data = {'x': X_train[0], 't': X_train[1],  'u1': u1_data, 'u2': u2_data}
        sio.savemat(path+'predict.mat', my_data)


        my_data = {'x': X_train[0], 't': X_train[1],  'u1': y_train[0], 'u2': y_train[1]}

        sio.savemat(path+'true.mat', my_data)


    elif method == "deeponet":

        df = pd.read_csv("matlab_3.csv")












        u1_data = df["u1"].values
        u2_data = df["u2"].values
        mae1 = np.mean(np.abs(u1_data - y_train[0]))
        var1 = np.var(u1_data - y_train[0])
        print(mae1, var1)
        mae2 = np.mean(np.abs(u2_data - y_train[1]))
        var2 = np.var(u2_data - y_train[1])
        print(mae2, var2)
        pass
    elif method == "physo":

        import pickle
        with open(path + 'sympy_expressions_u1.pkl', 'rb') as f:
            u1_exp = pickle.load(f)
        with open(path + 'sympy_expressions_u2.pkl', 'rb') as f:
            u2_exp = pickle.load(f)


        mase1_list = []
        for i in range(len(u1_exp)):
            u1_exp[i] = str2ind(str(u1_exp[i]), toolbox)
            u1_data = Instance1_onedata(toolbox, X_train[0], X_train[1], u1_exp[i])
            mae1 = np.mean(np.abs(u1_data - y_train[0]))
            mase1_list.append(mae1)



        min_value = min(mase1_list)

        u1_i = mase1_list.index(min_value)

        mase2_list = []
        for i in range(len(u2_exp)):
            u2_exp[i] = str2ind(str(u2_exp[i]), toolbox)
            u2_data = Instance1_onedata(toolbox, X_train[0], X_train[1], u2_exp[i])
            mae1 = np.mean(np.abs(u2_data - y_train[1]))
            mase2_list.append(mae1)


        min_value = min(mase2_list)

        u2_i = mase2_list.index(min_value)
        u1_data, u2_data = Instance1_data(toolbox, X_train[0], X_train[1], u1_exp[u1_i], u2_exp[u2_i])
        mae1 = np.mean(np.abs(u1_data - y_train[0]))
        var1 = np.var(u1_data - y_train[0])
        print(mae1, var1)
        mae2 = np.mean(np.abs(u2_data - y_train[1]))
        var2 = np.var(u2_data - y_train[1])
        print(mae2, var2)

        error = Instance1_PDEs(toolbox, X_train[0], X_train[1], u1_exp[u1_i], u2_exp[u2_i],path)
        print(error)









if __name__ == "__main__":
    paths = ["data", "data_instance_2", "data_instance_3"]


    main("data_instance_2",method="T-Baseline")