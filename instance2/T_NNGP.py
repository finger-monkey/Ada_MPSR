





import math
import copy
from deap import base, creator, gp, tools, algorithms
import operator

import random
import numpy as np
from local_optimize import *
from sympy import symbols,expand
import sympy
from parse_string import *
from sympy.parsing.sympy_parser import parse_expr
from utils import *
import copy
import pickle

def define_gp(fit_weight=(-1.0,)):


    pset = gp.PrimitiveSet("MAIN", 4)
    pset.renameArguments(ARG0='x')
    pset.renameArguments(ARG1='y')
    pset.renameArguments(ARG2='z')
    pset.renameArguments(ARG3='t')



    pset.addPrimitive(sympy.Mul, 2)
    pset.addPrimitive(sympy.Add, 2)
    pset.addPrimitive(operator.sub, 2)

    pset.addPrimitive(sympy.sin, 1)


    pset.addTerminal(terminal=1.0)




    creator.create("FitnessMin", base.Fitness, weights=fit_weight)


    creator.create("Individual", gp.PrimitiveTree,
                   fitness=creator.FitnessMin,
                   params=None,
                   local_optimize=local_optimize)






    toolbox = base.Toolbox()









    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_= 5 )


    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)


    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("compile", gp.compile, pset=pset)





    toolbox.register("select", tools.selNSGA2)




    toolbox.register("leafmate", gp.cxOnePointLeafBiased, termpb=0.1)


    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))


    toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


    def from_string(expr_str, pset):
        tree = gp.PrimitiveTree.from_string(expr_str, pset)
        return tools.initIterate(creator.Individual, lambda: tree)

    toolbox.register("individual_from_string", from_string, pset=pset)



    def from_list(expr_list):
        tree = gp.PrimitiveTree(expr_list)
        return tools.initIterate(creator.Individual, lambda: tree);

    toolbox.register("individual_list", from_list)

    return toolbox,pset



def str2ind(str1,toolbox):
    """
    sol_1d = convert_power(str1)
    sol_1d = parse_expr(sol_1d, evaluate=False)
    prefix_str = to_prefix(sol_1d)
    ind_1d = toolbox.individual_from_string(prefix_str)
    return ind_1d

def prob_mapping(values):
    """
    min_val = min(values)
    max_val = max(values)

    values = [(val - min_val) / (max_val - min_val+0.1) for val in values]

    values = [1 - val+0.1 for val in values]

    total = sum(values)
    weights = [val / total for val in values]
    return weights
def T_Baseline(knowledge_base, weight_u, X_train,y, toolbox,pset,match_path) :

    popnum = 500
    ngen = 100
    cxpb = 0.1
    mutpb = 0.1



    best_inds = []

    pop_u1 = []
    pop_u2 = []
    pop = [pop_u1,pop_u2]



    prob_u = prob_mapping(weight_u)

    knowledge_base_index = [ x for x in range(len(prob_u))]

    sample_size = popnum//len(knowledge_base[0])+1

    offspring = [[],[]]
    coords = copy.deepcopy(X_train)
    y_train = copy.deepcopy(y)
    index = [i for i in range(len(coords[0]))]
    for g in range(ngen):
        sample = np.random.choice(index, size=100, replace=False)
        X_train = coords[:, sample[:]]
        y[0] = y_train[0][sample[:]]
        y[1] = y_train[1][sample[:]]


        decouple_order = []
        pop_u1 = []
        pop_u2 = []
        pop = [pop_u1, pop_u2]

        for i in range(sample_size):
            sample = np.random.choice(knowledge_base_index, size=len(prob_u), replace=False, p=prob_u)
            temp_size = 1

            target_length = len(knowledge_base[0])

            count = 0
            for j in sample:
                count += 1
                if count == len(sample):
                    selected_lists = knowledge_base[j]
                else:
                    selected_lists = random.sample(knowledge_base[j], temp_size)
                    if temp_size < target_length:
                        temp_size = temp_size * 2

                while len(selected_lists) < target_length:
                    selected_lists.extend(copy.deepcopy(selected_lists))

                if len(selected_lists) > target_length:
                    selected_lists = selected_lists[:target_length]
                pop[j].extend(copy.deepcopy(selected_lists))
            decouple_order.extend([copy.deepcopy(sample) for _ in range(len(selected_lists))])

        temp = copy.deepcopy(offspring)
        pop = [pop[i] + temp[i] for i in range(len(pop))]


        for i in range(len(pop[0])):
            try:
                pop[0][i].local_optimize(X_train, y[0], pset)
                pop[1][i].local_optimize(X_train, y[1], pset)
                pop[0][i].fitness.values = Instance2_PDEs(toolbox, X_train[0], X_train[1], X_train[2], X_train[3], pop[0][i], pop[1][i],match_path)
                if np.isnan(pop[0][i].fitness.values[0]):
                    pop[0][i].fitness.values = [1000]
                pop[1][i].fitness.values = copy.deepcopy(pop[0][i].fitness.values)
            except:

                pop[0][i].fitness.values = [1000]
                pop[1][i].fitness.values = copy.deepcopy(pop[0][i].fitness.values)





        offspring = copy.deepcopy(pop)

        weight_u = list(map(lambda x: x*0.1 % 100, weight_u))
        for i in range(len(decouple_order)):



            temp = 1
            for j in decouple_order[i]:
                weight_u[j] += weight_u[j] + 1/(temp)*pop[0][i].fitness.values[0]
                temp+=1

            if random.random() < mutpb:
                index_pop = decouple_order[i][-1]
                mutant = offspring[index_pop][i]
                if len(mutant)<=2:
                    continue
                toolbox.mutate(mutant)
                del mutant.fitness.values


            if random.random() < cxpb:
                index_pop = decouple_order[i][0]
                offspring[index_pop][i] = copy.deepcopy(random.choice(knowledge_base[index_pop]))
                del offspring[index_pop][i].fitness.values

        prob_u = prob_mapping(weight_u)


        for i in range(len(offspring[0])):
            if not offspring[0][i].fitness.valid or not offspring[0][i].fitness.valid:
                try:
                    offspring[0][i].local_optimize(X_train, y[0], pset)
                    offspring[1][i].local_optimize(X_train, y[1], pset)

                    offspring[0][i].fitness.values = Instance2_PDEs(toolbox,X_train[0], X_train[1], X_train[2], X_train[3], offspring[0][i], offspring[1][i],match_path)
                    if np.isnan(offspring[0][i].fitness.values[0]):
                        offspring[0][i].fitness.values = [1000]
                    offspring[1][i].fitness.values = copy.deepcopy(offspring[0][i].fitness.values)
                except:
                    offspring[0][i].fitness.values = [1000]
                    offspring[1][i].fitness.values = copy.deepcopy(offspring[0][i].fitness.values)




        pop = [pop[i]+offspring[i] for i in range(len(pop))]
        pop[:] = pop + offspring
        fitness_set = set()
        new_pops = [[],[]]
        for i,ind in enumerate(pop[0]):
            if ind.fitness not in fitness_set:
                fitness_set.add(ind.fitness)
                new_pops[0].append(pop[0][i])
                new_pops[1].append(pop[1][i])
        pop = copy.deepcopy(new_pops)

        selected_individuals = toolbox.select(pop[0], popnum)

        selected_indices = [i for i, ind in enumerate(pop[0]) if ind in selected_individuals]




        offspring = [[],[]]

        for i in selected_indices:
            offspring[0].append(pop[0][i])
            offspring[1].append(pop[1][i])






        best_ind_u1 = min(offspring[0], key=lambda ind: ind.fitness.values[0])
        best_ind_u2 = min(offspring[1], key=lambda ind: ind.fitness.values[0])


        best_ind.local_optimize(X_train, y, pset)

        best_ind.fitness.values = toolbox.evaluate(ind)
        best_inds.append(copy.deepcopy([best_ind_u1,best_ind_u2]))







        x0, y0,z0,t0 = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('t')

        expr_u1 = toolbox.compile(expr=best_ind_u1)
        expr_u1 = expr_u1(x0, y0, z0,t0)
        expr_u1= expand(expr_u1)

        expr_u2 = toolbox.compile(expr=best_ind_u2)
        expr_u2 = expr_u2(x0, y0, z0,t0)
        expr_u2 = expand(expr_u2)

        print(f"Generation {g}: {str(expr_u1),str(expr_u2),best_ind_u1.fitness.values}")
        if best_ind_u1.fitness.values[0] < 1e-3:
            break


def main(path):


    x, y, z, t, u1, u2 = read_data(path)


    coords = np.vstack([x, y, z, t])
    X_train = coords
    y_train = [u1, u2]

    toolbox, pset = define_gp()

    with open(path + 'sympy_expressions_u1.pkl', 'rb') as f:
        u1_exp = pickle.load(f)
    with open(path + 'sympy_expressions_u2.pkl', 'rb') as f:
        u2_exp  = pickle.load(f)








    for i in range(len(u1_exp)):
        u1_exp[i] = str2ind(str(u1_exp[i]), toolbox)
    for i in range(len(u2_exp)):
        u2_exp[i] = str2ind(str(u2_exp[i]), toolbox)

    weight_u = [4, 3]
    knowledge_base = (u1_exp, u2_exp)
    T_Baseline(knowledge_base, weight_u, X_train, y_train, toolbox, pset, path)






if __name__ == '__main__':
    paths = ["comsol_instance1", "comsol_instance2", "comsol_instance3", "comsol_instance4", "comsol_instance5"]
    main("comsol_instance1")
