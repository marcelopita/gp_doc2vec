#!/bin/bash

input_dataset=$1
output_vectors=$2
vectors_size=$3

echo ""
date
echo ""

time ./paragraph2vec.py -i $input_dataset -o $output_vectors -s $vectors_size
