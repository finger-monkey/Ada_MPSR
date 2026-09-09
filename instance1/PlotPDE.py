






import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

def plot3D(x,y,z,u):



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')










    u = 0.000583*z**2 - 0.409*z + 1.07


    sc = ax.scatter(x, y, z, c=u, cmap='viridis')



    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    fig.colorbar(sc)

    plt.savefig("instance1_u2_pred.png")


    plt.show()

if __name__ == '__main__':
    path = "cs"
    x, y, z, u1, u2 = read_data(path)
    plot3D(x, y, z, u2)