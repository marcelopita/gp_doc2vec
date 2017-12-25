#!/bin/bash

./run_clusters.sh ./words_vectors/wv_sg_only-ds-100x_50.txt ./pkls/gp_doc2vec_clusters_20ng_sg_50_2016-11-22.pkl ./logs/gp_doc2vec_clusters_20ng_sg_50_2016-11-22.log ./data/20ng/20ng-train.csv ./data/20ng/20ng-test.csv 15 5 1 0.7 0.3 10 7 0.3 0.2 one_point_lb macro
