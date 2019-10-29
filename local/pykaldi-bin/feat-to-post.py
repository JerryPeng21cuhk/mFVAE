#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2019 Jerry Peng
#  mimics kaldi feat-to-post

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
# from deco import concurrent, synchronized
from kaldi.hmm import Posterior
from kaldi.util.table import SequentialMatrixReader, PosteriorWriter
# from utils import ConfigBase
import numpy as np
import pdb
# import os


def feat_to_post(feature_rspecifier, posterior_wspecifier, top_n=10):
  with SequentialMatrixReader(feature_rspecifier) as feature_reader, \
          PosteriorWriter(posterior_wspecifier) as posterior_writer:
    for uttid, feat in feature_reader:
      feat_np = feat.numpy()
      posts_lst = []
      assert top_n <= feat_np.shape[1]
      for row in feat_np:
        idxs = np.argpartition(row, -top_n)[-top_n:]
        post = [(int(idx), float(row[idx])) for idx in idxs]
        posts_lst.append(post)

      posterior_writer[uttid] = Posterior().from_posteriors(posts_lst)
  return True


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Convert features into posterior format, which is the generic format
  of NN training target in Karel's nnet1 tools.
  (spped is not an issue for reasonably low NN-output dimensions)
  Usage: feat-to-post.py [options] feature_rspecifier posteriors_wspecifier

  e.g.
      feat-to-post scp:feats.scp ark:post.ark

  """
  po = ParseOptions(usage)
  po.register_int("top-n", 10,
                     "N posteriors per frame, 10 by default")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  feature_rspecifier = po.get_arg(1)
  posterior_wspecifier = po.get_arg(2)
  isSuccess = feat_to_post(feature_rspecifier, posterior_wspecifier,
                           opts.top_n)
  if not isSuccess:
    sys.exit()

