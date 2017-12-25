#!/bin/bash

#words_dim=$1
docs_dim=$1
words_vectors_file=$2
pickle_output_file=$3
log_output_file=$4
centroids_file=$5
train_data=$6
test_data=$7
num_threads=$8
population_size=$9
hof_size=${10}
crossover_rate=${11}
mutation_rate=${12}
num_iters=${13}
max_tree_depth=${14}
validation_fraction=${15}
elitism_fraction=${16}
crossover_method=${17} # one_point or one_point_lb
log_reg_mode=${18} # binary or macro

echo ""
date
echo ""

time ./src/gp_doc2vec.py -dd $docs_dim -wv $words_vectors_file -o $pickle_output_file -ol $log_output_file -tr $train_data -te $test_data -th $num_threads -ps $population_size -hs $hof_size -cr $crossover_rate -mr $mutation_rate -i $num_iters -md $max_tree_depth -vf $validation_fraction -ef $elitism_fraction -cm $crossover_method -lrm $log_reg_mode -ce $centroids_file
