#!/home/jerry/anaconda3/envs/pykaldi/bin/python

# 2019 Jerry Peng
# This script converts varaible length of feature sequences into
# the same length by zero-padding.
# The output of this script will be fed into resnet34

from __future__ import print_function, division

import sys
import logging

from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialMatrixReader, MatrixWriter
import numpy as np


def pad_feats_edge(feature_rspecifier, feature_wspecifier, num_frame=1):
  with SequentialMatrixReader(feature_rspecifier) as feature_reader, \
          MatrixWriter(feature_wspecifier) as feature_writer:
    assert num_frame > 0  # num_frame <= 0 makes no sense
    num_utts_done = 0
    for uttid, feat in feature_reader:
      frame_num = feat.shape[0]
      feat_np = feat.numpy()
      feature_writer[uttid] = np.pad(feat_np, ((num_frame, num_frame), (0, 0)), 'edge')
      num_utts_done += 1
  logging.info("Done {} utterances".format(num_utts_done))
  return True if num_utts_done != 0 else False


if __name__ == '__main__':
  # Configure log messages to look like Kaldi messages
  from kaldi import __version__
  logging.addLevelName(20, "LOG")
  logging.basicConfig(format="%(levelname)s (%(module)s[{}]:%(funcName)s():"
                             "%(filename)s:%(lineno)s) %(message)s"
                             .format(__version__), level=logging.INFO)
  usage = """Padding frames at both edges by duplicating frames at edges 
  with a specified number
  Usage: pad-feats-edge.py [options] feature_rspecifier feature_wspecifier

  e.g.
      pad-feats-edge.py scp:feats.scp ark,scp:exp/feats_pad.ark/feats_pad.scp,

  """
  po = ParseOptions(usage)
  po.register_int("num-frame", 1,
                  "the number of frames for padding, 1 by default")
  opts = po.parse_args()

  if (po.num_args() != 2):
    po.print_usage()
    sys.exit()

  feature_rspecifier = po.get_arg(1)
  feature_wspecifier = po.get_arg(2)
  isSuccess = pad_feats_edge(feature_rspecifier, feature_wspecifier,
                             num_frame=opts.num_frame)
  if not isSuccess:
    sys.exit()

