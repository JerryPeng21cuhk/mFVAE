#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2019 Jerry Peng
# This script computes mean and variance given input features.
# It mimics kaldi executable: compute-cmvn-stats

from __future__ import print_function, division

import sys
import logging

from collections import defaultdict

from kaldi.transform.cmvn import Cmvn # acc_cmvn_stats, acc_cmvn_stats_single_frame
from kaldi.util.options import ParseOptions
# from deco import concurrent, synchronized
from kaldi.util.table import SequentialMatrixReader, MatrixWriter
# from utils import ConfigBase
import numpy as np
# import os


def compute_global_cmvn_stats(feature_rspecifier, stats_wxfilename, binary):
  with SequentialMatrixReader(feature_rspecifier) as feature_reader:
    num_utts_done, num_err = 0, 0
    is_init = False
    for uttid, feat in feature_reader:
      if not is_init:
        feat_dim = feat.num_cols
        cmvn = Cmvn(feat_dim)
      else:
        assert feat_dim == feat.num_cols, "feature dimension ({}) of utterance {} doesn't match {}".format(feat.num_cols, uttid, feat_dim)
      cmvn.accumulate(feat)
      num_utts_done += 1

    cmvn.write_stats(stats_wxfilename, binary)

  logging.info("Done {} utterances".format(num_utts_done))
  return True if num_utts_done != 0 else False


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Compute global CMVN stats for a data set
  Usage: compute-global-cmvn-stats.py [options] feature_rspecifier stats_wxfilename

  e.g.
      compute-global-cmvn-stats.py scp:data/train/feats.scp data/train/cmvn_stats,

  """
  po = ParseOptions(usage)
  po.register_bool("binary", False,
                     "write CMVN stats in binary format. False by default")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  feature_rspecifier = po.get_arg(1)
  stats_wxfilename = po.get_arg(2)
  isSuccess = compute_global_cmvn_stats(feature_rspecifier, stats_wxfilename, opts.binary)
  if not isSuccess:
    sys.exit()

