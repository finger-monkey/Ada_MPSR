




from T_Baseline import *
def main(path,method = "T-Baseline"):


    x, y, z, t, u1, u2 = read_data(path)


    coords = np.vstack([x, y, z, t])
    X_train = coords
    y_train = [u1, u2]

    toolbox, pset = define_gp()

    if method == "T-Baseline":
























        ind1 = str2ind("1000.01925481801*x*sin(y) - 1000.01925481801*x + 0.999985552268344", toolbox)
        ind2 = str2ind("y*z**2 - z**3 + 368.65937593336", toolbox)



        u1_data, u2_data = Instance2_data(toolbox, X_train[0], X_train[1], X_train[2], X_train[3], ind1, ind2)
        mae1 = np.mean(np.abs(u1_data - y_train[0]))
        var1 = np.var(u1_data - y_train[0])
        print(mae1, var1)
        mae2 = np.mean(np.abs(u2_data - y_train[1]))
        var2 = np.var(u2_data - y_train[1])
        print(mae2, var2)
        error = Instance2_PDEs(toolbox,X_train[0], X_train[1], X_train[2], X_train[3], ind1, ind2,path)
        print(error)

    elif method == "deeponet":

        df = pd.read_csv("comsol_5.csv")



        u1_data = df["V"].values
        u2_data = df["T"].values
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
            u1_data = Instance2_onedata(toolbox, X_train[0], X_train[1],X_train[2], X_train[3], u1_exp[i])
            mae1 = np.mean(np.abs(u1_data - y_train[0]))
            mase1_list.append(mae1)


        min_value = min(mase1_list)

        u1_i = mase1_list.index(min_value)

        mase2_list = []
        for i in range(len(u2_exp)):
            u2_exp[i] = str2ind(str(u2_exp[i]), toolbox)
            u2_data = Instance2_onedata(toolbox, X_train[0], X_train[1],X_train[2], X_train[3], u2_exp[i])
            mae1 = np.mean(np.abs(u2_data - y_train[1]))
            mase2_list.append(mae1)


        min_value = min(mase2_list)

        u2_i = mase2_list.index(min_value)
        u1_data, u2_data = Instance2_data(toolbox, X_train[0], X_train[1],X_train[2], X_train[3], u1_exp[u1_i], u2_exp[u2_i])
        mae1 = np.mean(np.abs(u1_data - y_train[0]))
        var1 = np.var(u1_data - y_train[0])
        print(mae1, var1)
        mae2 = np.mean(np.abs(u2_data - y_train[1]))
        var2 = np.var(u2_data - y_train[1])
        print(mae2, var2)

        error = Instance2_PDEs(toolbox, X_train[0], X_train[1],X_train[2], X_train[3], u1_exp[u1_i], u2_exp[u2_i],path)
        print(error)









if __name__ == '__main__':
    paths = ["comsol_instance1", "comsol_instance2", "comsol_instance3", "comsol_instance4", "comsol_instance5"]
    main("comsol_instance5",method="deeponet")