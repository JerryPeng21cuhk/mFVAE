#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2019 Herman
# 2019 Jerry Peng

# This script use t-sne to convert high-dimension vectors into low-dimension space.

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialVectorReader, VectorWriter, read_script_file,\
                            classify_rspecifier, RspecifierType,\
                            classify_wspecifier, WspecifierType
import numpy as np

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Added for runing on the server side
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import pdb


def tsne_vector(vector_rspecifier, vector_wspecifier, output_dim=2, perplexity=30, learning_rate=200.0, n_iter=1000, distance='euclidean', verbose=0):
  vectors = []
  with SequentialVectorReader(vector_rspecifier) as vector_reader:
    for uttid, vector in vector_reader:
      vectors.append(vector.numpy())
  # vectors is a set of row vectors indexed by utterance id
  vectors = np.array(vectors)
  tsne = TSNE(n_components=output_dim, perplexity=perplexity, learning_rate=learning_rate, metric=distance, verbose=verbose)
  low_dim_vectors = tsne.fit_transform(vectors)  ## return a numpy array of row vectors indexed by utterance id
  with SequentialVectorReader(vector_rspecifier) as vector_reader, \
        VectorWriter(vector_wspecifier) as vector_writer:
    for i, (uttid, _) in enumerate(vector_reader):
      vector_writer[uttid] = low_dim_vectors[i]
  return True

if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Use t-sne (t-distributed Stochastic Neighbor Emedding) for dimension reduction.
  For the details, Please refer to website:
  https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE

  Usage: tsne-vector.py [options] <vector-rspecifier> <vector-wspecifier

  e.g.
      tsne-vector.py scp:data/train/ivector.scp ark:data/train/low_dim_vector.ark
  """
  po = ParseOptions(usage)
  po.register_int("output-dim", 2,
                  "dimension of the output vectors."
                  " For visualization, only 2 is allowed in this program. (2 by default)")
  po.register_double("perplexity", 30,
                     "The perplexity is related to the number of nearest neighbors that is used"
                     " in other mainfold learning algorithms. Large datasets usually require a"
                     " large perplexity. Consider selecting a value between 5 and 50. Different"
                     " values can result in significantly different results. (30 by default)")
  po.register_double("learning-rate", 200.0,
                     "The learning rate for t-sne is usually in the range [10.0, 1000.0]. If the"
                     " learning rate is too high, the data may look like a \'ball\' with any point"
                     " approximately equidistant from its nearest neighbors. If the learning rate"
                     " is too low, most points may look compressed in a dense cloud with few outliers."
                     " If the cost function gets stuck in a bad local minimum increasing the learning"
                     " rate may help. (200.0 by default)")
  po.register_int("n-iter", 1000,
                  "Maximum number of iterations for the optimization. Should be at least 250. (1000 by default)")
  po.register_str("distance", "euclidean",
                  "The distance measurement between vectors. \n"
                  "It could be [euclidean|cityblock|mahalanobis|cosine|...]. For more options, please refer to"
                  " https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html"
                  " (euclidean by default)")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  vector_rspecifier = po.get_arg(1)
  vector_wspecifier = po.get_arg(2)
  pdb.set_trace()
  isSuccess = tsne_vector(vector_rspecifier, 
                          vector_wspecifier, 
                          output_dim=opts.output_dim, 
                          perplexity=opts.perplexity, 
                          learning_rate=opts.learning_rate, 
                          n_iter=opts.n_iter, 
                          distance=opts.distance, 
                          verbose=1)
  if not isSuccess:
    sys.exit()
  