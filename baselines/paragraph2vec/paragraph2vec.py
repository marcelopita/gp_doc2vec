#!/usr/bin/python -W ignore
# -*- coding: utf-8 -*-

import sys
import random
import gensim
import argparse

argv = sys.argv
parser = argparse.ArgumentParser(description = "Generates docs vectors using Paragraph2Vec.")
parser.add_argument("-i", "--input_docs_filename", help = "Input documents file", required = True)
parser.add_argument("-o", "--output_docs_vectors_filename", help = "Output documents vectors file", required = True)
parser.add_argument("-s", "--docs_vectors_size", help = "Documents vectors size", type = int, default = 50)
parser.add_argument("-w", "--window_size", help = "Window size", type = int, default = 10)
parser.add_argument("-mc", "--min_count", help = "Minimum frequency of words", type = int, default = 1)
parser.add_argument("-t", "--num_threads", help = "Number of threads", type = int, default = 20)
parser.add_argument("-ni", "--num_iterations", help = "Number of iterations", type = int, default = 10)
args = parser.parse_args(argv[1:])

print "Reading documents...",
sys.stdout.flush()
docs = gensim.models.doc2vec.TaggedLineDocument(args.input_docs_filename)
print "OK!"

print "Learning model...",
sys.stdout.flush()
model = gensim.models.doc2vec.Doc2Vec(docs, size = args.docs_vectors_size, window = args.window_size,
                                      min_count = args.min_count, workers = args.num_threads,
                                      iter = args.num_iterations, sample = 1e-3, negative = 5)
print "OK!"

print "Saving model to disk...",
sys.stdout.flush()
model.save_word2vec_format(args.output_docs_vectors_filename)
print "OK!"
sys.stdout.flush()
