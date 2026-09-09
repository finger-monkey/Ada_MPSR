

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import minimize
from collections import Counter
import math

SEED = 87
random.seed(SEED)
np.random.seed(SEED)





file_path = "./simulated_joule_heating_data.csv"
df = pd.read_csv(file_path)

x_data = df["x"].values
y_data = df["y"].values
u_data = df["u"].values
T_data = df["T"].values

                                
                                               
def evaluate_expression(params, x, y, u):
    a, b, c, d = params
    return a * u**2 + b * x + c * y + d

                        
def compute_mae(params, x, y, u, T_true):
    T_pred = evaluate_expression(params, x, y, u)
    return np.mean(np.abs(T_pred - T_true))

                  
def compute_entropy(op_pool):
    counter = Counter(op_pool)
    total = sum(counter.values())
    probs = [count / total for count in counter.values()]
    return -sum(p * math.log(p + 1e-8) for p in probs)

                                       
def normalize_entropy_series(entropy_series):
    H_min = min(entropy_series)
    H_max = max(entropy_series)
    return [(h - H_min) / (H_max - H_min + 1e-8) for h in entropy_series]

                       
def init_population(size):
    population = []
    for _ in range(size):
        a, b, c, d = np.random.randn(4)
        ops = random.choices(['+', '*', 'u^2', 'x', 'y'], k=4)
        population.append({'params': [a, b, c, d], 'ops': ops})
    return population

                   
def mutate(individual, strength=0.1):
    params = individual['params']
    new_params = [p + strength * np.random.randn() for p in params]
    ops = deepcopy(individual['ops'])
    if random.random() < 0.25:
        idx = np.random.randint(0, len(ops))
        ops[idx] = random.choice(['+', '*', 'u^2', 'x', 'y'])
    return {'params': new_params, 'ops': ops}

                                
               
def run_experiment(mutation_mode="entropy", fixed_rate=0.3, iterations=60, pop_size=20):
    pop_entropy = init_population(pop_size)
    pop_fixed = deepcopy(pop_entropy)

    history_mae_entropy, history_entropy_entropy = [], []
    history_mae_fixed, history_entropy_fixed = [], []

    for t in range(iterations):
                                  
        errors_e, new_pop_e, op_pool_e = [], [], []

        for indiv in pop_entropy:
            mae = compute_mae(indiv['params'], x_data, y_data, u_data, T_data)
            errors_e.append(mae)
            op_pool_e += indiv['ops']

        H_e = compute_entropy(op_pool_e)
        norm_H_e = (H_e - 0.0) / (2.0 + 1e-8)
        mutation_rate = 1 - norm_H_e

        for indiv in pop_entropy:
            if random.random() < mutation_rate:
                child = mutate(indiv)
            else:
                child = deepcopy(indiv)
            new_pop_e.append(child)

        pop_entropy = new_pop_e
        history_mae_entropy.append(np.mean(errors_e))
        history_entropy_entropy.append(H_e)

                                  
        errors_f, new_pop_f, op_pool_f = [], [], []

        for indiv in pop_fixed:
            mae = compute_mae(indiv['params'], x_data, y_data, u_data, T_data)
            errors_f.append(mae)
            op_pool_f += indiv['ops']

        H_f = compute_entropy(op_pool_f)

        for indiv in pop_fixed:
            if random.random() < fixed_rate:
                child = mutate(indiv)
            else:
                child = deepcopy(indiv)
            new_pop_f.append(child)

        pop_fixed = new_pop_f
        history_mae_fixed.append(np.mean(errors_f))
        history_entropy_fixed.append(H_f)

                              
    norm_entropy_entropy = normalize_entropy_series(history_entropy_entropy)
    norm_entropy_fixed = normalize_entropy_series(history_entropy_fixed)

    return history_mae_entropy, norm_entropy_entropy, history_mae_fixed, norm_entropy_fixed

              
mae_entropy, norm_ent_entropy, mae_fixed, norm_ent_fixed = run_experiment()

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(np.array(mae_entropy), label="Entropy-Guided")
axs[0].plot(np.array(mae_fixed), label="Fixed Mutation (p=0.3)", linestyle='--')
axs[0].set_title("MAE over Iterations")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("MAE (x1e-3)")
axs[0].legend()

axs[1].plot(norm_ent_entropy, label="Entropy-Guided")
axs[1].plot(norm_ent_fixed, label="Fixed Mutation (p=0.3)", linestyle='--')
axs[1].set_title("Normalized Structural Entropy")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Entropy (normalized)")
axs[1].legend()

plt.tight_layout()
plt.show()
