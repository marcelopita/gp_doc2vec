#!/usr/bin/python -W ignore
# -*- coding: utf-8 -*-

'''
File: gp_doc2vec_clusters.py

Generates documents vectors from words vectors through genetic programming.
Words vectors of a document are first clusterized, than uniform functions to
be discovered by the GP algorithm are applied to clusters, generating at the
end a document vector of dimensionality D (same as words vectors). Therefore,
if you want documents vectors of dimensionlity D, you have to produce first
words vectors of dimensionality D.

Author: Marcelo Pita
Created: 2016/11/15

Modified: 2016/11/15 (Marcelo Pita) (First version)

'''


# Imports
import sys
import argparse
import gensim
from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg
import operator
import random
from deap import base, creator, tools, algorithms, gp
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from multiprocessing import Pool
import pickle
import math


# Random seed
random.seed(997)


# Global var vectors dimensionality
D = 0


'''
Load words vectors model
'''
def load_words_vectors(wv_filename, is_wv_file_bin):
    words_model = None
    try:
        words_model = gensim.models.Word2Vec.load_word2vec_format(wv_filename, binary=is_wv_file_bin)
    except:
        print "Unable to open words vectors file: " + wv_filename
    return words_model


'''
Cluster class.
'''
class WordsVectorsCluster(object):

    def __init__(self):
        self.words_vectors = []
        self.all_pairs_distance_sum = 0.0

    def insert_word_vector(self, word_vector):
        self.words_vectors.append(word_vector)
        for wv in self.words_vectors:
            self.all_pairs_distance_sum += np.linalg.norm(wv - word_vector)

    def mean_intracluster_distance(self):
        num_elems = len(self.words_vectors)
        if num_elems == 0:
            return 0.0
        elif num_elems == 1:
            return 1.0
        num_pairs = num_elems * (num_elems - 1) / 2.0
        return self.all_pairs_distance_sum / num_pairs


'''
Infer document cluster
'''
def infer_doc_clusters(doc_words_vectors):
    X = np.array(doc_words_vectors)

    clusters = [WordsVectorsCluster(), WordsVectorsCluster(), WordsVectorsCluster()]

    if len(X) < 3:
        for i in range(0, len(X)):
            clusters[0].insert_word_vector(doc_words_vectors[i])
            clusters[1].insert_word_vector(doc_words_vectors[i])
            clusters[2].insert_word_vector(doc_words_vectors[i])
        return clusters

    kmeans_model = KMeans(n_clusters = 3, random_state = 0).fit(X)

    for i in range(0, len(X)):
        clusters[kmeans_model.labels_[i]].insert_word_vector(doc_words_vectors[i])

    # Sort clusters by intra-cluster distance
    clusters.sort(key=operator.methodcaller("mean_intracluster_distance"), reverse=False)

    return clusters


'''
Get documents clusters
'''
def get_docs_clusters(wv_model, docs_filename, wv_cache, shuffle = False):
    # Open docs file
    docs_file = None
    try:
        docs_file = open(docs_filename, 'r')
    except:
        print "Unable to open documents file: " + docs_filename
        return None

    # Docs clusters list
    docs_clusters = []
    docs_classes = []

    # For each document d
    for d in docs_file:
        id_class_text = d.split(';')

        # Retrieve words of d
        words = set(id_class_text[2].split())
        if len(words) == 0:
            continue

        # List of words vectors for d
        doc_words_vectors = []

        # For each word w
        for w in words:
            # Put word vector for w into cache, if it exists
            if not (w in wv_cache):
                wv = None
                try:
                    wv = wv_model[w]
                except:
                    continue
                wv_cache[w] = wv
            # Append wv to d words vectors list
            doc_words_vectors.append(wv_cache[w])

        # If d words_vectors list is empty, next.
        if not doc_words_vectors:
            continue

        # Store doc class and clusters
        docs_classes.append(id_class_text[1])
        docs_clusters.append(infer_doc_clusters(doc_words_vectors))

    # Optional shuffling
    if shuffle:
        docs_indexes = range(0, len(docs_classes))
        random.shuffle(docs_indexes)
        docs_classes = [docs_classes[i] for i in docs_indexes]
        docs_clusters = [docs_clusters[i] for i in docs_indexes]

    return (docs_classes, docs_clusters)


### GP TYPES, OPERATORS, FUNCTIONS, ... ###



class Type1(object):
    def __init__(self, v):
        self.vec = v

class Type2(object):
    def __init__(self, v):
        self.vec = v

class Type3(object):
    def __init__(self, v):
        self.vec = v

class Type4(object):
    def __init__(self, v):
        self.vec = v



def adjust_vector(v):
    return [float(x) for x in np.nan_to_num(v)]

def add_type1(cluster):
    result = None
    for wv in cluster.words_vectors:
        if result is None:
            result = wv
            continue
        result = adjust_vector([x+y for x,y in zip(result, wv)])
#        result = list(adjust_vector(map(operator.add, result, wv)))
    return Type1(result)

def add_type3(p1, p2, p3):
    v1 = p1.vec
    v2 = p2.vec
    v3 = p3.vec

    if v1 is None:
        v1 = D * [0.0]
    if v2 is None:
        v2 = D * [0.0]
    if v3 is None:
        v3 = D * [0.0]
    return Type3(adjust_vector([x+y+z for x,y,z in zip(v1, v2, v3)]))
#    return list(adjust_vector(map(operator.add, list(adjust_vector(map(operator.add, v1, v2))), v3)))

def add_external2(v1, v2):
    if v1 is None:
        v1 = D * [0.0]
    if v2 is None:
        v2 = D * [0.0]
    return adjust_vector([x+y for x,y in zip(v1, v2)])
#    return list(adjust_vector(map(operator.add, v1, v2)))

def sub_type1(cluster):
    result = None
    for wv in cluster.words_vectors:
        if result is None:
            result = wv
            continue
        result = adjust_vector([x-y for x,y in zip(result, wv)])
#        result = list(adjust_vector(map(operator.sub, result, wv)))
    return Type1(result)

def sub_type3(p1, p2, p3):
    v1 = p1.vec
    v2 = p2.vec
    v3 = p3.vec

    if v1 is None:
        v1 = D * [0.0]
        if v2 is not None:
            v2 = map(operator.neg, v2)
        elif v3 is not None:
            v3 = map(operator.neg, v3)
    if v2 is None:
        v2 = D * [0.0]
        if v3 is not None:
            v3 = map(operator.neg, v3)
    if v3 is None:
        v3 = D * [0.0]
    return Type3(adjust_vector([x-y-z for x,y,z in zip(v1, v2, v3)]))
#    return list(adjust_vector(map(operator.sub, list(adjust_vector(map(operator.sub, v1, v2))), v3)))

def sub_external2(v1, v2):
    if v1 is None:
        v1 = D * [0.0]
        if v2 is not None:
            v2 = map(operator.neg, v2)
    if v2 is None:
        v2 = D * [0.0]
    return adjust_vector([x-y for x,y in zip(v1, v2)])
#    return list(adjust_vector(map(operator.sub, v1, v2)))

def mul_type1(cluster):
    result = None
    for wv in cluster.words_vectors:
        if result is None:
            result = wv
            continue
        result = adjust_vector([x*y for x,y in zip(result,wv)])
#        result = list(map(operator.mul, result, wv))
    return Type1(result)

def mul_type2(s, t1):
    v = t1.vec
    if v is None:
        v = D * [1.0]
    return Type2(list(adjust_vector([x*s for x in v])))

def mul_type4(s, t3):
    v = t3.vec
    if v is None:
        v = D * [1.0]
    return Type4(list(adjust_vector([x*s for x in v])))

def mul_type3(p1, p2, p3):
    v1 = p1.vec
    v2 = p2.vec
    v3 = p3.vec

    if v1 is None:
        v1 = D * [1.0]
    if v2 is None:
        v2 = D * [1.0]
    if v3 is None:
        v3 = D * [1.0]
    return Type3(adjust_vector([x*y*z for x,y,z in zip(v1,v2,v3)]))
#    return list(adjust_vector(map(operator.mul, list(adjust_vector(map(operator.mul, v1, v2))), v3)))

def mul_external2(v1, v2):
    if v1 is None:
        v1 = D * [1.0]
    if v2 is None:
        v2 = D * [1.0]
    return adjust_vector([x*y for x,y in zip(v1,v2)])
#    return list(adjust_vector(map(operator.mul, v1, v2)))

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

def div_type1(cluster):
    result = None
    for wv in cluster.words_vectors:
        if result is None:
            result = wv
            continue
        result = adjust_vector([protectedDiv(x,y) for x,y in zip(result,wv)])
#        result = list(adjust_vector(map(protectedDiv, result, wv)))
    return Type1(result)

def div_type3(p1, p2, p3):
    v1 = p1.vec
    v2 = p2.vec
    v3 = p3.vec

    if v1 is None:
        if v2 is not None:
            v1 = map(lambda x : x*x, v2)
        elif v3 is not None:
            v1 = v3
            v2 = map(lambda x: 1.0/x, v3)
        else:
            v1 = D * [1.0]
            v2 = D * [1.0]
            v3 = D * [1.0]
    if v2 is None:
        v2 = D * [1.0]
    if v3 is None:
        v3 = D * [1.0]
    return Type3(adjust_vector([protectedDiv(protectedDiv(x,y),z) for x,y,z in zip(v1,v2,v3)]))
#    return div_external2(v3, div_external2(v1, v2))
#    return list(adjust_vector(map(protectedDiv, list(adjust_vector(map(protectedDiv, v1, v2))), v3)))

def div_external2(v1, v2):
    if v1 is None:
        if v2 is not None:
            v1 = map(lambda x : x*x, v2)
        else:
            v1 = D * [1.0]
            v2 = D * [1.0]
    if v2 is None:
        v2 = D * [1.0]
    return adjust_vector([protectedDiv(x,y) for x,y in zip(v1,v2)])
#    return list(adjust_vector(map(protectedDiv, v1, v2)))

def norm_logistic(v):
    if v is None:
        return None
    v_exp = adjust_vector(np.exp(map(operator.neg, v)))
    return [(0.0001 + 1 / (1 + s)) for s in v_exp]

def norm_unit(v):
    if v is None:
        return None
    magnitude = 0.0
    for s in v:
        magnitude += s*s
    magnitude = math.sqrt(magnitude)
    return list(adjust_vector([s/magnitude for s in v]))

def softmax(v):
    if v is None:
        return None
    v_exp = list(adjust_vector(np.exp(v)))
    return (v_exp / np.sum(v_exp, axis=0))

def pow_type1(cluster):
    result = None
    for wv in cluster.words_vectors:
        if result is None:
            result = norm_logistic(wv)
            continue
        result = adjust_vector([operator.pow(x,y) for x,y in zip(result,norm_logistic(wv))])
#        result = list(adjust_vector(map(operator.pow, result, norm_logistic(wv))))
    return Type1(result)

def pow_type3(p1, p2, p3):
    v1 = p1.vec
    v2 = p2.vec
    v3 = p3.vec

    if v1 is None:
        v1 = D * [1.0]
    if v2 is None:
        v2 = D * [1.0]
    if v3 is None:
        v3 = D * [1.0]
    v1 = norm_logistic(v1)
    v2 = norm_logistic(v2)
    v3 = norm_logistic(v3)
    return Type3(adjust_vector([operator.pow(operator.pow(x,y),z) for x,y,z in zip(v1,v2,v3)]))
#    return list(adjust_vector(map(operator.pow, list(adjust_vector(map(operator.pow, v1, v2))), v3)))

def ident_final(t4):
    return list(t4.vec)

def my_pow_external2(v1, v2):
    if v1 is None:
        v1 = D * [1.0]
    if v2 is None:
        v2 = D * [1.0]
    v1 = norm_logistic(v1)
    v2 = norm_logistic(v2)
    return adjust_vector([operator.pow(x,y) for x,y in zip(v1,v2)])
#    return list(adjust_vector(map(operator.pow, v1, v2)))

def generate_random_number():
    return random.uniform(-1.0, 1.0)


def evalClassif(individual, toolbox, logistic_regression_mode, train_X, valid_X, train_Y, valid_Y):
    func = toolbox.compile(expr=individual)

#    # Sampling 50%
#    sample_idxs = random.sample(train_indexes, indiv_train_sample_size)
#
#    # Training and validation datasets
#    train_X, valid_X, train_Y, valid_Y = train_test_split([train_docs_clusters[i] for i in sample_idxs],
#                                                          [train_docs_classes[i] for i in sample_idxs],
#                                                          test_size = validation_fraction,
#                                                          random_state = 0)

    print "Representing docs...",
    sys.stdout.flush()

    # Documents vectors
    train_documents_vectors = []
    for tr_dc in train_X:
        train_documents_vectors.append(func(*tr_dc))
    valid_documents_vectors = []
    for va_dc in valid_X:
        valid_documents_vectors.append(func(*va_dc))

    print "OK!"
    sys.stdout.flush()


    print "Training classifier...",
    sys.stdout.flush()

    # Classifier training and validation
    classif = LogisticRegression()
    classif.fit(train_documents_vectors, train_Y)
    valid_pred_Y = classif.predict(valid_documents_vectors)
    fitness = 0.0
    if logistic_regression_mode == "macro":
        fitness = f1_score(valid_Y, valid_pred_Y, average = logistic_regression_mode)
    elif logistic_regression_mode == "binary":
        fitness = f1_score(valid_Y, valid_pred_Y, average = logistic_regression_mode, pos_label='2')

    print "OK!"
    sys.stdout.flush()

    return fitness,


def eaElitism(population, toolbox, N, cxpb, mutpb, ngen, stats,
              halloffame, verbose, train_docs_clusters, train_docs_classes,
              validation_fraction, logistic_regression_mode):

    train_indexes = range(0,len(train_docs_classes))
    indiv_train_sample_size = len(train_docs_classes)/2

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Sampling 50%                                                                                                              
    sample_idxs = random.sample(train_indexes, indiv_train_sample_size)
    # Training and validation datasets                                                                                          
    train_X, valid_X, train_Y, valid_Y = train_test_split([train_docs_clusters[i] for i in sample_idxs],
                                                          [train_docs_classes[i] for i in sample_idxs],
                                                          test_size = validation_fraction,
                                                          random_state = 0)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = [toolbox.evaluate(i, toolbox, logistic_regression_mode, train_X, valid_X, train_Y, valid_Y)
                 for i in invalid_ind]

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
        print "Geracao: " + str(gen)
        sys.stdout.flush()

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

        # Sampling 50%                                                     
        sample_idxs = random.sample(train_indexes, indiv_train_sample_size)
        # Training and validation datasets
        train_X, valid_X, train_Y, valid_Y = train_test_split([train_docs_clusters[i] for i in sample_idxs],
                                                              [train_docs_classes[i] for i in sample_idxs],
                                                              test_size = validation_fraction,
                                                              random_state = 0)

        fitnesses = [toolbox.evaluate(i, toolbox, logistic_regression_mode, train_X, valid_X, train_Y, valid_Y)
                 for i in invalid_ind]
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


def create_primitive_set(vectors_dimension):
    pset = gp.PrimitiveSetTyped("gp_doc2vec_clusters",
                                [WordsVectorsCluster, WordsVectorsCluster, WordsVectorsCluster], list, "x")

    pset.addEphemeralConstant("k", generate_random_number, float)
#    pset.addEphemeralConstant("k2", generate_random_number, float)
#    pset.addEphemeralConstant("k3", generate_random_number, float)
#    pset.addEphemeralConstant("k4", generate_random_number, float)
    pset.addEphemeralConstant("l", lambda: [generate_random_number() for i in range(0, vectors_dimension)], list)
#    pset.addEphemeralConstant("t1", lambda: Type1([generate_random_number() for i in range(0, vectors_dimension)]), Type1)
#    pset.addEphemeralConstant("t2", lambda: Type2([generate_random_number() for i in range(0, vectors_dimension)]), Type2)
#    pset.addEphemeralConstant("t3", lambda: Type3([generate_random_number() for i in range(0, vectors_dimension)]), Type3)
#    pset.addEphemeralConstant("t4", lambda: Type4([generate_random_number() for i in range(0, vectors_dimension)]), Type4)

    pset.addTerminal(Type1([generate_random_number() for i in range(0, vectors_dimension)]), Type1)
    pset.addTerminal(Type2([generate_random_number() for i in range(0, vectors_dimension)]), Type2)
    pset.addTerminal(Type3([generate_random_number() for i in range(0, vectors_dimension)]), Type3)
    pset.addTerminal(Type4([generate_random_number() for i in range(0, vectors_dimension)]), Type4)

    pset.addPrimitive(add_type1, [WordsVectorsCluster], Type1)
    pset.addPrimitive(sub_type1, [WordsVectorsCluster], Type1)
    pset.addPrimitive(mul_type1, [WordsVectorsCluster], Type1)
    pset.addPrimitive(div_type1, [WordsVectorsCluster], Type1)
    pset.addPrimitive(pow_type1, [WordsVectorsCluster], Type1)

    pset.addPrimitive(mul_type2, [float, Type1], Type2)

    pset.addPrimitive(add_type3, [Type2, Type2, Type2], Type3)
    pset.addPrimitive(sub_type3, [Type2, Type2, Type2], Type3)
    pset.addPrimitive(mul_type3, [Type2, Type2, Type2], Type3)
    pset.addPrimitive(div_type3, [Type2, Type2, Type2], Type3)
    pset.addPrimitive(pow_type3, [Type2, Type2, Type2], Type3)

    pset.addPrimitive(mul_type4, [float, Type3], Type4)

    pset.addPrimitive(ident_final, [Type4], list)


#    pset.addPrimitive(add, [WordsVectorsCluster], list)
#    pset.addPrimitive(add_external1, [list, list, list], list)
#    pset.addPrimitive(add_external2, [list, list], list)
#
#    pset.addPrimitive(sub, [WordsVectorsCluster], list)
#    pset.addPrimitive(sub_external1, [list, list, list], list)
#    pset.addPrimitive(sub_external2, [list, list], list)
#
#    pset.addPrimitive(mul, [WordsVectorsCluster], list)
#    pset.addPrimitive(mul_scalar, [list, float], list)
#    pset.addPrimitive(mul_external1, [list, list, list], list)
#    pset.addPrimitive(mul_external2, [list, list], list)
#
#    pset.addPrimitive(div, [WordsVectorsCluster], list)
#    pset.addPrimitive(div_external1, [list, list, list], list)
#    pset.addPrimitive(div_external2, [list, list], list)
#
#    pset.addPrimitive(my_pow, [WordsVectorsCluster], list)
#    pset.addPrimitive(my_pow_external1, [list, list, list], list)
#    pset.addPrimitive(my_pow_external2, [list, list], list)
#
#    pset.addPrimitive(norm_logistic, [list], list)
#    pset.addPrimitive(norm_unit, [list], list)
#    pset.addPrimitive(softmax, [list], list)
#
    pset.addPrimitive(lambda x:x, [float], float, name="ident_float")
    pset.addPrimitive(lambda x:x, [WordsVectorsCluster], WordsVectorsCluster, name="ident_wv_cluster")

    return pset


def create_toolbox(pset, max_depth, crossover_method):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
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
    return toolbox
    

def run_gp(toolbox, num_threads, population_size, hof_size, elite_fraction, crossover_rate, mutation_rate, max_iterations,
           train_docs_clusters, train_docs_classes, validation_fraction, logistic_regression_mode):
    pool = Pool(num_threads)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n = population_size)
    hof = tools.HallOfFame(hof_size)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = eaElitism(pop, toolbox, elite_fraction, crossover_rate, mutation_rate,
                         max_iterations, mstats, hof, True, train_docs_clusters,
                         train_docs_classes, validation_fraction, logistic_regression_mode)

    pool.close()

    return dict(population=pop, log=log, hof=hof)


def test_classifier(toolbox, best_individual, wv_model, test_filename, wv_cache,
                    train_docs_clusters, train_docs_classes, logistic_regression_mode):
    func = toolbox.compile(expr = best_individual)

    test_docs_classes, test_docs_clusters = get_docs_clusters(wv_model, test_filename, wv_cache)

    test_documents_vectors = []
    for te_dc in test_docs_clusters:
        test_documents_vectors.append(func(*te_dc))

    train_documents_vectors = []
    for tr_dc in train_docs_clusters:
        train_documents_vectors.append(func(*tr_dc))

    classif = LogisticRegression()
    classif.fit(train_documents_vectors, train_docs_classes)
    test_pred_Y = classif.predict(test_documents_vectors)

    test_score = 0.0
    if logistic_regression_mode == "macro":
        test_score = f1_score(test_docs_classes, test_pred_Y, average = logistic_regression_mode)
    elif logistic_regression_mode == "binary":
        test_score = f1_score(test_docs_classes, test_pred_Y, average = logistic_regression_mode, pos_label='2')

    return test_score
    
                     

########################


def main(argv=None):
    # Parsing command line arguments
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(description = "Generates docs vectors from words vectors through genetic programming.")
    parser.add_argument("-wv", "--wv_filename", help = "Words vectors file", required = True)
    parser.add_argument("-wvb", "--is_wv_file_bin", help="Set words vectors file to binary",
                        action='store_true', default=False)
    parser.add_argument("-o", "--output_filename", help="Output file (dump of objects)", default="output.pkl")
    parser.add_argument("-ol", "--output_log_filename", help="Log file", default="output.log")
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
    args = parser.parse_args(argv[1:])

    # Load words vectors model
    print "Loading words vectors...",
    sys.stdout.flush()
    wv_model = load_words_vectors(args.wv_filename, args.is_wv_file_bin)
    print "OK!"

    # Set vectors dimensionality
    global D
    D = wv_model.vector_size

    # Words vectors cache
    wv_cache = dict()

    # Get train documents clusters
    print "Loading training data...",
    sys.stdout.flush()

    train_docs_classes = None
    train_docs_clusters = None

    try:
        data_dump_file = open("/tmp/gp_doc2vec_clusters_data.tmp", 'r')
        train_docs_classes, train_docs_clusters = pickle.load(data_dump_file)
        data_dump_file.close()
    except:
        data_dump_file = open("/tmp/gp_doc2vec_clusters_data.tmp", 'w')
        train_docs_classes, train_docs_clusters = get_docs_clusters(wv_model, args.train_filename, wv_cache, shuffle = True)
        pickle.dump((train_docs_classes, train_docs_clusters), data_dump_file, 2)
        data_dump_file.close()

    print "OK!"

    # GP primitive set
    pset = create_primitive_set(wv_model.vector_size)

    # Create fitness and individual functions
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    # Toolbox
    toolbox = create_toolbox(pset, args.max_depth, args.crossover_method)

    # Run GP
    results_gp = run_gp(toolbox, args.num_threads, args.population_size, args.hof_size, args.elite_fraction,
                        args.crossover_rate, args.mutation_rate, args.max_iterations, train_docs_clusters,
                        train_docs_classes, args.validation_fraction, args.logistic_regression_mode)

    # Save results
    dump_file = open(args.output_filename, 'w')
    pickle.dump(results_gp, dump_file, 2)
    dump_file.close()

    # Test representation
    test_score = test_classifier(toolbox, results_gp['hof'][0], wv_model, args.test_filename, wv_cache,
                                 train_docs_clusters, train_docs_classes, args.logistic_regression_mode)

    # Write log to disk
    log_file = open(args.output_log_filename, 'w')
    log_file.write(results_gp['log'] + "\n\n" + "F1 score: " + str(test_score))
    log_file.close()
    
    
if __name__ == '__main__':
    main()
