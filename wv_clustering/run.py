import sys, os
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

wv_path = sys.argv[1]
base_path = sys.argv[2]
cluster_outpath = sys.argv[3]
scale = float(sys.argv[4])

basename = os.path.basename(base_path)
vnew_name = os.path.dirname(base_path) + '/' + \
            basename.split('.')[0] + '.vocabulary'


# if vocabulary file is already stored
if os.path.isfile(vnew_name):
  print "Reading vocabulary..."
  v_in = open(vnew_name)
  vocab = set([line.strip() for line in v_in])
else:
  print "Creating vocabulary..."
  v_in = open(base_path)
  vocab = set()
  for line in v_in:
    words = line.split(';')[2].strip().split()
    for w in words:
      vocab.add(w)

  print "Storing vocabulary..."
  v_out = open(vnew_name, 'w')
  for w in vocab:
    v_out.write(w + '\n')


print "Loading vectors..."
wv_model = Word2Vec.load_word2vec_format(wv_path, binary=False)
dim = wv_model.layer1_size
X = []
for word in vocab:
  X.append(wv_model[word])


print "Clustering word vectors..."
k = int(scale * len(vocab))
kmeans = KMeans(n_clusters=k, 
                precompute_distances=True, 
                n_jobs=-2,
                random_state=42).fit(X)


print "Storing centroids found..."
fout = open(cluster_outpath, 'w')
for cluster in kmeans.cluster_centers_:
  str_c = "".join([str(i) + ' ' for i in cluster])
  fout.write(str_c + '\n')
fout.close()





