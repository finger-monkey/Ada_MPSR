





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

def plot3D(x,y,z,u,t):



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')














    sc = ax.scatter(x, y, z, c=u, cmap='viridis')



    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    fig.colorbar(sc)

    plt.savefig(f"instance2_1_u2_pred_t_{t}.png")


    plt.show()

if __name__ == '__main__':
    paths = ["comsol_instance1", "comsol_instance2", "comsol_instance3", "comsol_instance4", "comsol_instance5"]
    time = 0.5
    x, y, z, t, u1, u2 = read_data(paths[0])
    indices = np.where(t == time)
    plot3D(x[indices], y[indices], z[indices], u2[indices],time)