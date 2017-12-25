#!/bin/bash

./run.sh 50 ./words_vectors/wv_sg_only-ds-100x_50.txt ./pkls/gp_doc2vec_20ng_sg_50-50_2016-10-31.pkl ./logs/gp_doc2vec_20ng_sg_50-50_2016-10-31.log ./wv_clustering/ng_clusters_001.txt ./data/20ng/20ng-train.csv ./data/20ng/20ng-test.csv 40 150 1 0.7 0.3 150 5 0.3 0.1 one_point_lb macro
