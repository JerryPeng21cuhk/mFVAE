#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2019 Jerry Peng
# This script computes the mean of given speaker vectors

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialVectorReader
from kaldi.util.io import write_vector
from kaldi.matrix import Vector
import numpy as np


def vector_mean(vector_rspecifier, vector_wxfilename, binary=False):
  with SequentialVectorReader(vector_rspecifier) as vector_reader:
    num_utts_done = 0
    for uttid, vector in vector_reader:
      # frame_num = feat.shape[0]
      vec_np = vector.numpy()
      if num_utts_done == 0:
        vec_sum = vec_np
      else:
        vec_sum = vec_sum + vec_np
      num_utts_done += 1
  logging.info("Done {} utterances".format(num_utts_done))
  vec_mean = vec_sum / num_utts_done
  write_vector(Vector(vec_mean), vector_wxfilename, binary=binary)
  return True if num_utts_done != 0 else False


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Average vectors (e.g. speaker vectors)
  Usage: vector-mean.py [options] vector_rspecifier vector_wxfilename
  e.g.
      vector-mean.py scp:vector.scp mean.vec,
  """
  po = ParseOptions(usage)
  po.register_bool("binary", False,
                  "write output as binary, False by default")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  vector_rspecifier = po.get_arg(1)
  vector_wxfilename = po.get_arg(2)
  isSuccess = vector_mean(vector_rspecifier, vector_wxfilename,
                             binary=opts.binary)
  if not isSuccess:
    sys.exit()

