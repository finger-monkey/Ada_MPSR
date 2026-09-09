import numpy as np
import pandas as pd
import random
import math
from collections import Counter, defaultdict
from copy import deepcopy

                 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

                                    
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

                     
def compute_entropy(op_pool):
    counter = Counter(op_pool)
    total = sum(counter.values())
    probs = [count / total for count in counter.values()]
    return -sum(p * math.log(p + 1e-8) for p in probs)

                               
def analyze_expression_diversity(population):
    op_pool = []
    expr_counts = defaultdict(int)
    for indiv in population:
        op_signature = tuple(indiv["ops"])
        op_pool.extend(op_signature)
        expr_counts[op_signature] += 1
    entropy = compute_entropy(op_pool)
    total = sum(expr_counts.values())
    max_duplicate_ratio = max(expr_counts.values()) / total
    unique_ratio = len(expr_counts) / total
    return entropy, max_duplicate_ratio, unique_ratio

                           
def one_step_mutate(population, mode="entropy", fixed_rate=0.3):
    mutated = []
    op_pool = []
    for indiv in population:
        op_pool += indiv["ops"]
    H = compute_entropy(op_pool)
    norm_H = (H - 0.0) / (2.0 + 1e-8)
    mutation_rate = 1 - norm_H if mode == "entropy" else fixed_rate
    for indiv in population:
        if random.random() < mutation_rate:
            child = mutate(indiv)
        else:
            child = deepcopy(indiv)
        mutated.append(child)
    return mutated

                        
pop_entropy = init_population(30)
pop_fixed = deepcopy(pop_entropy)

             
mutated_entropy = one_step_mutate(pop_entropy, mode="entropy")
mutated_fixed = one_step_mutate(pop_fixed, mode="fixed", fixed_rate=0.3)

                   
diversity_entropy = analyze_expression_diversity(mutated_entropy)
diversity_fixed = analyze_expression_diversity(mutated_fixed)

diversity_entropy, diversity_fixed
