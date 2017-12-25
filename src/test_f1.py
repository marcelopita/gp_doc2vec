#!/usr/bin/python -W ignore
# -*- coding: utf-8 -*-

import numpy as np
import sys
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from deap import base, creator, tools, algorithms, gp
import pickle
from gp_operators import *

test_filename = "./data/20ng/20ng-test.csv"
train_filename = "./data/20ng/20ng-train.csv"
words_vectors_filename = "./words_vectors/wv_sg_only-ds-100x_50.txt"
centroids_filename = "./wv_clustering/clusters_001.txt"
output_log_filename = "./logs/gp_doc2vec_20ng_sg_50-50.log"
dump_filename = "./pkls/gp_doc2vec_20ng_sg_50-50.pkl"
is_words_vectors_file_binary = False
D2 = 50
max_depth = 5
crossover_method = "one_point_lb"
logistic_regression_mode = "macro"

words_model = gensim.models.Word2Vec.load_word2vec_format(words_vectors_filename, binary=is_words_vectors_file_binary)

print "Loading words vectors centroids...",
sys.stdout.flush()
centroids_file = open(centroids_filename, 'r')
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

func = toolbox.compile(expr=results_gp['hof'][0])

deltas_cache = dict()

train_classes = []
train_input_vectors = []
train_file = open(train_filename, 'r')
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

test_classes = []
test_input_vectors = []
test_file = open(test_filename, 'r')
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
if logistic_regression_mode == "macro":
    test_score = f1_score(test_classes, test_pred_Y, average=logistic_regression_mode)
elif logistic_regression_mode == "binary":
    test_score = f1_score(test_classes, test_pred_Y, average=logistic_regression_mode, pos_label='2')

print "\nF1 score: " + str(test_score)

log_file = open(output_log_filename, 'w')
log_file.write(str(results_gp['log']) + "\n\n" + "F1 score: " + str(test_score))
log_file.close()
