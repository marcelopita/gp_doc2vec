#!/bin/bash

words_vectors_file=$1
pickle_output_file=$2
log_output_file=$3
train_data=$4
test_data=$5
num_threads=$6
population_size=$7
hof_size=$8
crossover_rate=$9
mutation_rate=${10}
num_iters=${11}
max_tree_depth=${12}
validation_fraction=${13}
elitism_fraction=${14}
crossover_method=${15} # one_point or one_point_lb
log_reg_mode=${16} # binary or macro

echo ""
date
echo ""

time ./src/gp_doc2vec_clusters/gp_doc2vec_clusters.py -wv $words_vectors_file -o $pickle_output_file -ol $log_output_file -tr $train_data -te $test_data -th $num_threads -ps $population_size -hs $hof_size -cr $crossover_rate -mr $mutation_rate -i $num_iters -md $max_tree_depth -vf $validation_fraction -ef $elitism_fraction -cm $crossover_method -lrm $log_reg_mode
