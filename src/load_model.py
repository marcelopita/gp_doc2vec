from deap import base, creator, tools, algorithms, gp
import pickle
from gp_operators import *

# A definir: num_centroids, D2, max_depth, crossover_method, dump_filename, logistic_regression_mode

def generate_random_number():
    return random.uniform(-1.0, 1.0)

def generate_random_tuple():
    return D2 * (generate_random_number(),)

pset = gp.PrimitiveSetTyped("evo_doc2vec", num_centroids*[list], tuple, "x")

pset.addEphemeralConstant("k1", generate_random_number, float)
pset.addEphemeralConstant("k2",  generate_random_tuple, tuple)

pset.addPrimitive(ident, D2*[float], tuple)
pset.addPrimitive(ident_unit, D2*[float], tuple)
pset.addPrimitive(ident_softmax, D2*[float], tuple)
pset.addPrimitive(neg, [list], list)
pset.addPrimitive(add1, [list, list], list)
pset.addPrimitive(add2, [list, float], list)
pset.addPrimitive(sub1, [list, list], list)
pset.addPrimitive(sub2, [list, float], list)
pset.addPrimitive(mul1, [list, list], list)
pset.addPrimitive(mul2, [list, list], float)
pset.addPrimitive(mul3, [list, float], list)

pset.addPrimitive(div1, [list, list], list)
pset.addPrimitive(div2, [list, float], list)
pset.addPrimitive(pow1, [list, list], list)
pset.addPrimitive(pow2, [list, float], list)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalClassif(individual):
    func = toolbox.compile(expr=individual)
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_input_vectors, train_classes,
                                                          test_size = args.validation_fraction,
                                                          random_state = 0)
    train_documents_vectors = []
    for tr_iv in train_X:
        train_documents_vectors.append(list(func(*tr_iv)))
    valid_documents_vectors = []
    for va_iv in valid_X:
        valid_documents_vectors.append(list(func(*va_iv)))
    classif = LogisticRegression()
    classif.fit(train_documents_vectors, train_Y)
    valid_pred_Y = classif.predict(valid_documents_vectors)
    fitness = 0.0
    if logistic_regression_mode == "macro":
        fitness = f1_score(valid_Y, valid_pred_Y, average=logistic_regression_mode)
    elif logistic_regression_mode == "binary":
        fitness = f1_score(valid_Y, valid_pred_Y, average=logistic_regression_mode, pos_label='2')
    return fitness,

toolbox.register("evaluate", evalClassif)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("selectElite", tools.selBest)
if crossover_method == "one_point":
    toolbox.register("mate", gp.cxOnePoint)
elif crossover_method == "one_point_lb":
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

dump_file = open(dump_filename, 'r')
results_gp = pickle.load(dump_file)
dump_file.close()
