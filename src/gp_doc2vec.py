#!/usr/bin/python -W ignore
# -*- coding: utf-8 -*-


'''

File: gp_doc2vec.py

Generates documents vectors from words vectors through genetic programming.

Author: Marcelo Pita
Created: 2016/09/21

Modified: 2016/09/21 (Marcelo Pita)    (First version)
Modified: 2016/10/05 (Marcelo Pita)    (Unit vectors operator, softmax operator,
                                        dataset format, not found words vectors)
Modified: 2016/10/14 (Marcelo Pita)    (Logistic regression with macro and binary options)
Modified: 2016/10/20 (Marcelo Pita,
                      Gabriel Pedrosa) (Consider clusters of words vectors (centroids),
                                        documents matrices based on centroids)
Modified: 2016/10/31 (Marcelo Pita)    (Training sampling 50%)

'''


# Imports
import sys
# sys.path.append("/home/gabriel/anaconda2/lib/python2.7/site-packages")
from deap import base, creator, tools, algorithms, gp
import math
import array
import random
import operator
import gensim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from multiprocessing import Pool
import pickle
import argparse
from gp_operators import *


# Arguments parsing

argv = sys.argv
parser = argparse.ArgumentParser(description = "Generates docs vectors from words vectors through genetic programming.")
#parser.add_argument("-dw", "--dim_words_vectors", help="Dimensionality of words vectors", type=int, default=20)
parser.add_argument("-dd", "--dim_docs_vectors", help="Dimensionality of documents vectors", type=int, default=20)
parser.add_argument("-wv", "--words_vectors_filename", help="Words vectors file", required=True)
parser.add_argument("-wvb", "--is_words_vectors_file_binary", help="Set words vectors file to binary",
                    action='store_true', default=False)
parser.add_argument("-o", "--output_filename", help="Output file (dump of objects)", default="output.pkl")
parser.add_argument("-ol", "--output_log_filename", help="Log file", default="output.log")
#parser.add_argument("-ftr", "--full_train_filename", help="Full train file", required=True)
parser.add_argument("-tr", "--train_filename", help="Train file", required=True)
parser.add_argument("-te", "--test_filename", help="Test file", required=True)
parser.add_argument("-th", "--num_threads", help="Number of threads", type=int, default=8)
parser.add_argument("-ps", "--population_size", help="GP population size", type=int, default=150)
parser.add_argument("-hs", "--hof_size", help="Hall of fame size", type=int, default=1)
parser.add_argument("-cr", "--crossover_rate", help="Crossover rate", type=float, default=0.7)
parser.add_argument("-mr", "--mutation_rate", help="Mutation rate", type=float, default=0.2)
parser.add_argument("-i", "--max_iterations", help="Max iterations", type=int, default=150)
parser.add_argument("-md", "--max_depth", help="Max tree depth", type=int, default=5)
parser.add_argument("-vf", "--validation_fraction", help="Validation fraction", type=float, default=0.3)
parser.add_argument("-ef", "--elite_fraction", help="Elite fraction", type=float, default=0.1)
parser.add_argument("-cm", "--crossover_method", help="Crossover method", default="one_point")
parser.add_argument("-lrm", "--logistic_regression_mode", help="Logist regression mode (macro ou binary)", default="macro")
parser.add_argument("-ce", "--centroids_filename", help="Centroids file", required=True)
args = parser.parse_args(argv[1:])

D2 = args.dim_docs_vectors


# Random seed
random.seed(997)

# Words model
print "Loading words vectors...",
sys.stdout.flush()
words_model = gensim.models.Word2Vec.load_word2vec_format(args.words_vectors_filename, binary=args.is_words_vectors_file_binary)
print "OK!"
sys.stdout.flush()


print "Loading words vectors centroids...",
sys.stdout.flush()
centroids_file = open(args.centroids_filename, 'r')
centroids = None
num_centroids = 0
for centroid_str in centroids_file:
    if centroids is None:
        centroids = np.fromstring(centroid_str, dtype=np.float64, sep=' ')
    else:
        centroids = np.vstack((centroids, np.fromstring(centroid_str, dtype=np.float64, sep=' ')))
    num_centroids += 1
print "OK!"
sys.stdout.flush()


# Load train data

deltas_cache = dict()

train_classes = []
train_input_vectors = []
train_file = open(args.train_filename, 'r')
print "Loading train data...",
sys.stdout.flush()
for train_line in train_file:
    doc_matrix = None
    num_words = 0.0
    id_class_text = train_line.split(';')
    words = set(id_class_text[2].split())
    if len(words)==0:
        continue
    for w in words:
        if not (w in deltas_cache):
            wv = None
            try:
                wv = words_model[w]
            except:
                continue
            deltas_cache[w] = wv - centroids
        if doc_matrix is None:
            doc_matrix = deltas_cache[w]
        else:
            doc_matrix += deltas_cache[w]
        num_words += 1.0
    if doc_matrix is None:
        continue
    doc_matrix /= num_words
    train_classes.append(id_class_text[1])
    train_input_vectors.append(map(list, doc_matrix))
print "OK"
sys.stdout.flush()
train_file.close()

indiv_train_sample_size = len(train_classes) / 2
print "Train sample size for individuals (50%): " + str(indiv_train_sample_size)
sys.stdout.flush()

train_indexes = range(0,len(train_classes))
random.shuffle(train_indexes)
train_classes = [train_classes[i] for i in train_indexes]
train_input_vectors = [train_input_vectors[i] for i in train_indexes]

train_indexes = range(0,len(train_classes))

# Primitive set

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

##pset.addPrimitive(add3, [float, float], float)
##pset.addPrimitive(sub3, [float, float], float)
##pset.addPrimitive(mul4, [float, float], float)
##pset.addPrimitive(div3, [float, float], float)
##pset.addPrimitive(pow3, [float, float], float)


# Creator

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)


# Toolbox

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=args.max_depth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# using one training-validation set tuple for the whole execution 
#train_X, valid_X, train_Y, valid_Y = train_test_split(train_input_vectors, train_classes,
#                                                      test_size = args.validation_fraction,
#                                                      random_state = 0)
 
def evalClassif(individual):
    func = toolbox.compile(expr=individual)

    # Sampling 50%
    sample_idxs = random.sample(train_indexes, indiv_train_sample_size)

    # Training and validation datasets
    train_X, valid_X, train_Y, valid_Y = train_test_split([train_input_vectors[i] for i in sample_idxs],
                                                          [train_classes[i] for i in sample_idxs],
                                                          test_size = args.validation_fraction,
                                                          random_state = 0)
    # Documents vectors
    train_documents_vectors = []
    for tr_iv in train_X:
        train_documents_vectors.append(list(func(*tr_iv)))
    valid_documents_vectors = []
    for va_iv in valid_X:
        valid_documents_vectors.append(list(func(*va_iv)))

    # Classifier training and validation
    classif = LogisticRegression()
    classif.fit(train_documents_vectors, train_Y)
    valid_pred_Y = classif.predict(valid_documents_vectors)
    fitness = 0.0
    if args.logistic_regression_mode == "macro":
        fitness = f1_score(valid_Y, valid_pred_Y, average=args.logistic_regression_mode)
    elif args.logistic_regression_mode == "binary":
        fitness = f1_score(valid_Y, valid_pred_Y, average=args.logistic_regression_mode, pos_label='2')

    return fitness,


toolbox.register("evaluate", evalClassif)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("selectElite", tools.selBest)
if args.crossover_method == "one_point":
    toolbox.register("mate", gp.cxOnePoint)
elif args.crossover_method == "one_point_lb":
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# Evolution function

def eaElitism(population, toolbox, N, cxpb, mutpb, ngen, stats=None,
              halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen + 1):

        pop_size = len(population)
        elite_size = int(N * pop_size)
        offspring_size = pop_size - elite_size

        # Select the next generation individuals
        elite = toolbox.selectElite(population, elite_size)
        offspring = toolbox.select(population, offspring_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        next_generation = elite + offspring

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(next_generation)

        # Replace the current population by the offspring
        population[:] = next_generation

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream
            sys.stdout.flush()

    return population, logbook


# Run GP

pool = Pool(args.num_threads)
toolbox.register("map", pool.map)

pop = toolbox.population(n = args.population_size)
hof = tools.HallOfFame(args.hof_size)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

#pop, log = algorithms.eaSimple(pop, toolbox, args.crossover_rate, args.mutation_rate,
#                               args.max_iterations, stats=mstats, halloffame=hof, verbose=True)

pop, log = eaElitism(pop, toolbox, args.elite_fraction, args.crossover_rate, args.mutation_rate,
                     args.max_iterations, stats=mstats, halloffame=hof, verbose=True)

pool.close()

results_gp = dict(population=pop, log=log, hof=hof)


# Save results               
dump_file = open(args.output_filename, 'w')
pickle.dump(results_gp, dump_file, 2)
dump_file.close()


# Test classifier

func = toolbox.compile(expr=hof[0])

test_classes = []
test_input_vectors = []
test_file = open(args.test_filename, 'r')
print "Loading test data...",
sys.stdout.flush()
for test_line in test_file:
    doc_matrix = None
    num_words = 0.0
    id_class_text = test_line.split(';')
    words = set(id_class_text[2].split())
    if len(words)==0:
        continue
    for w in words:
        if not (w in deltas_cache):
            wv = None
            try:
                wv = words_model[w]
            except:
                continue
            deltas_cache[w] = wv - centroids
        if doc_matrix is None:
            doc_matrix = deltas_cache[w]
        else:
            doc_matrix += deltas_cache[w]
        num_words += 1.0
    if doc_matrix is None:
        continue
    doc_matrix /= num_words
    test_classes.append(id_class_text[1])
    test_input_vectors.append(map(list, doc_matrix))
print "OK"
sys.stdout.flush()
test_file.close()

test_documents_vectors = []
for te_iv in test_input_vectors:
    test_documents_vectors.append(list(func(*te_iv)))

train_documents_vectors = []
for tr_iv in train_input_vectors:
    train_documents_vectors.append(list(func(*tr_iv)))

classif = LogisticRegression()
classif.fit(train_documents_vectors, train_classes)
test_pred_Y = classif.predict(test_documents_vectors)

test_score = 0.0
if args.logistic_regression_mode == "macro":
    test_score = f1_score(test_classes, test_pred_Y, average=args.logistic_regression_mode)
elif args.logistic_regression_mode == "binary":
    test_score = f1_score(test_classes, test_pred_Y, average=args.logistic_regression_mode, pos_label='2')

print "\nF1 score: " + str(test_score)

# Log file
log_file = open(args.output_log_filename, 'w')
log_file.write(results_gp['log'] + "\n\n" + "F1 score: " + str(test_score))
log_file.close()
