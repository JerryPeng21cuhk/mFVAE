#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2019 Jerry Peng
#  mimics kaldi feat-to-post

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
# from deco import concurrent, synchronized
from kaldi.hmm import Posterior
from kaldi.util.table import SequentialMatrixReader, PosteriorWriter, VectorWriter
from kaldi.matrix import Vector
# from utils import ConfigBase
import numpy as np
import pdb
# import os


# def feat_to_post(feature_rspecifier, posterior_wspecifier, top_n=10):
#   with SequentialMatrixReader(feature_rspecifier) as feature_reader, \
#           PosteriorWriter(posterior_wspecifier) as posterior_writer:
#     for uttid, feat in feature_reader:
#       feat_np = feat.numpy()
#       posts_lst = []
#       assert top_n <= feat_np.shape[1]
#       for row in feat_np:
#         idxs = np.argpartition(row, -top_n)[-top_n:]
#         post = [(int(idx), float(row[idx])) for idx in idxs]
#         posts_lst.append(post)
# 
#       posterior_writer[uttid] = Posterior().from_posteriors(posts_lst)
#   return True

def feat_to_count(feature_rspecifier, cnt_wspecifier, normalize=False, per_utt=False):
  with SequentialMatrixReader(feature_rspecifier) as feature_reader, \
          VectorWriter(cnt_wspecifier) as cnt_writer:
      if per_utt:
        for uttid, feat in feature_reader:
          cnt_writer[uttid] = Vector(feat.numpy().mean(axis=0))
      else:
        vec = 0
        num_done = 0
        for uttid, feat in feature_reader:
          vec = vec + feat.numpy().mean(axis=0)
          num_done = num_done + 1
        if normalize:
          vec = vec / num_done
        # post = zip(range(len(vec)), vec.tolist())
        # posterior_writer[str(num_done)] = Posterior().from_posteriors([post])
        cnt_writer[str(num_done)] = Vector(vec)
  return True

    

if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Compute the counts of posterior for each mixture. If --normalize=True, 
  will normalize the counts
  Usage: post-count.py [options] feature_rspecifier posteriors_wspecifier

  e.g.
      post-count scp:feats.scp ark,t:count.txt

  """
  po = ParseOptions(usage)
  po.register_bool("normalize", False, "normalize the counts, False by default")
  po.register_bool("per-utt", False, "Count per utterance, False by default")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  feature_rspecifier = po.get_arg(1)
  posterior_wspecifier = po.get_arg(2)
  isSuccess = feat_to_count(feature_rspecifier, posterior_wspecifier, normalize=opts.normalize, per_utt=opts.per_utt)
  if not isSuccess:
    sys.exit()

