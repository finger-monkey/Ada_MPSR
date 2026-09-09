




from utils import *

def plot2D(x,t,u):



    fig = plt.figure()
    ax = fig.add_subplot(111)




















    u = 0.765 * np.sin(x)



    sc = ax.contourf(X, T, u, cmap='viridis')







    ax.set_xlabel("X")
    ax.set_ylabel("T")





    fig.colorbar(sc)

    plt.savefig(f"instance3_2_u2_pred.png")


    plt.show()

if __name__ == '__main__':
    paths = ["data", "data_instance_2", "data_instance_3"]
    x, t, u1, u2 = read_data(paths[1])




    X, T = np.meshgrid(x, t)
    plot2D(X,T,u2)