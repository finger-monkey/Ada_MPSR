




import numpy as np
import matplotlib.pyplot as plt
def plot_convergence():
    fno = np.load("instance1_mean_nocoupling.npz")["fitness_best"]
    fit = np.load("instance1_mean.npz")["fitness_best"]
    gen = np.array(range(len(fno)))

    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(gen, fno, "b-", label="Mean Fitness,T-Baseline-no-decoupling")
    plt.plot(gen, fit, "r-", label="Mean Fitness,T-Baseline")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("full_generation.png")
    plt.show()

def plot_convergence2():
    fno = np.load("instance1_mean_nocoupling.npz")["fitness_best"]
    fit = np.load("instance1_mean.npz")["fitness_best"]
    gen = np.array(range(len(fno)))

    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 22})
    plt.plot(gen[1:], fno[1:], "b-", label="Mean Fitness,T-Baseline-no-decoupling")
    plt.plot(gen[1:], fit[1:], "r-", label="Mean Fitness,T-Baseline")
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("partial_generation.png")
    plt.show()


if __name__ == '__main__':
    plot_convergence2()
