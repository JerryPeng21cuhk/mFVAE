import pdb
import os
import sys
import subprocess, io, threading
import kaldi_io
import numpy as np
from pprint import pprint
from collections import OrderedDict
from datetime import datetime


class Logger(object):
  def __init__(self, opath2logfile):
    self.terminal = sys.stdout
    self.log = open(opath2logfile, "w")
    self.write('Time: {0}\n'.format(str(datetime.now())))

  def write(self, message):
      self.terminal.write(message)
      self.log.write(message)  

  def flush(self):
      #this flush method is needed for python 3 compatibility.
      #this handles the flush command by doing nothing.
      #you might want to specify some extra behavior here.
      pass    


class ConfigBase(object):
  def __new__(cls, *args, **kwargs):
    instance = object.__new__(cls)
    instance.__odict__ = OrderedDict()
    return instance

  def __setattr__(self, key, value):
    if key != '__odict__':
      self.__odict__[key] = value
    object.__setattr__(self, key, value)

  def print_args(self):
    """
    Print all configurations
    """
    print("[Configuration]")
    for key, value in self.__odict__.items():
      print('\'{0}\' : {1}'.format(key, value))
    print('')

  def parse_args(self):
    """
    Supports to pass arguments from command line
    """
    import argparse
    def str2bool(v):
      if v.lower() in ('true', 't'):
        return True
      elif v.lower() in ('false', 'f'):
        return False
      else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser()
    for key, value in self.__odict__.items():
      if bool == type(value):
        parser.add_argument('--'+key.replace("_", "-"), default=str(value), type=str2bool)
      else:
        parser.add_argument('--'+key.replace("_", "-"), default=value, type=type(value))
    args = parser.parse_args()
    args = vars(args)
    # update
    for key in self.__odict__:
      arg = args[key]
      self.__odict__[key] = arg
      object.__setattr__(self, key, arg)


def save_predict_result(predict_result, dataset, opath2result, normalize=True):
  """
    format result and save it to opath2result
    the output should be formated as follows.
    
    Kazak Kazak_F0101033 6.049543
    Tibet Kazak_F0101033 1.100619
    Uyghu Kazak_F0101033 5.243432
    ct Kazak_F0101033 -0.9423696
    id Kazak_F0101033 -3.326587
    ja Kazak_F0101033 -0.8206892
    ko Kazak_F0101033 -3.42079
    ru Kazak_F0101033 3.852252
    vi Kazak_F0101033 -2.77112
    zh Kazak_F0101033 -0.8504514
    ...
 
    Note that the first two columns should have exactly the same format as above.
    Otherwise, it will cause logical bugs when the result is further used in other bash sciprts.
  """
  # lang_list = ['Kazak', 'Tibet', 'Uyghu', 'ct', 'id', 'ja', 'ko', 'ru', 'vi', 'zh']
  # lang_len = len(lang_list)

  num_utts, num_spks = predict_result.shape
  if True==normalize:
    predict_result = softmax(predict_result, axis=1)
  assert dataset.num_utts == num_utts, "number of utts mismatches!"
  assert dataset.num_spks == num_spks, "number of spks mismatches!"
  print(">> Save score result to %s" %opath2result)
  with open(opath2result, 'w') as f:
    for utt_idx, uttid in enumerate(dataset.uttids):
      for lan_idx, lang in enumerate(dataset.int2lan.values()):
        f.write("%s %s %.6f\n" %(lang, uttid, predict_result[utt_idx, lan_idx]))


# ## deprecated
# def read_gmm_model(model_rxfilename):
#   from kaldi.util import io as _util_io
#   from kaldi.gmm import FullGmm as fgmm
#   with _util_io.xopen(model_rxfilename) as ki:
#     fgmm = fgmm() # Full-covar GMM
#     fgmm.read(ki.stream(), ki.binary)
#   return fgmm


# def save_utt_embeddings(embeddings, dataset, oembed_dir):
#   from kaldi.util.table import VectorWriter
#   if not os.path.exists(oembed_dir):
#     os.makedirs(oembed_dir)
#   # embed_arkscp = 'ark:| copy-vector ark:- ark,scp:{0}/embeddings.ark,{0}/embeddings.scp'.format(oembed_dir)
#   embed_arkscp = 'ark,scp:{0}/embeddings.ark,{0}/embeddings.scp'.format(oembed_dir)
#   assert len(dataset.uttids) == embeddings.shape[0], "#embeddings({0}) doesn't match dataset #uttids{1}".format(embeddings.shape[0], len(dataset.uttids))
#   with VectorWriter(embed_arkscp) as vector_writer:
#     for utt_idx, uttid in enumerate(dataset.uttids):
#       vector_writer[uttid] = embeddings[utt_idx]
#   # with kaldi_io.open_or_fd(embed_arkscp, 'wb') as f:
#   #   for utt_idx, uttid in enumerate(dataset.uttids):
#   #     kaldi_io.write_vec_flt(f, embeddings[utt_idx], key=uttid)
#   print(">> Saved utterance embeddings to %s" %embed_arkscp)
# 
# 
# def save_spk_embeddings(ipath2uttembed_scp, ipath2spk2utt,  oembed_dir):
#   if not os.path.exists(oembed_dir):
#     os.makedirs(oembed_dir)
#   def cleanup(proc, cmd):
#     ret = proc.wait()
#     if ret > 0:
#       raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
#     return
#   # compute speaker-level embeddings
#   # spkembed_arkscp = "\'ark:| copy-vector ark:- ark,scp:{0}/spk_embeddings.ark,{0}/spk_embeddings.scp\'".format(oembed_dir)
#   spkembed_arkscp = "ark,scp:{0}/spk_embeddings.ark,{0}/spk_embeddings.scp".format(oembed_dir)
#   cmd = "ivector-mean ark:{0} scp:{1} {2} ark,t:{3}/num_utts.ark".format(ipath2spk2utt, ipath2uttembed_scp,
#     spkembed_arkscp, oembed_dir)
#   proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
#   thread = threading.Thread(target=cleanup, args=(proc, cmd))
#   thread.start()
#   thread.join() # wait the thread finish
#   print(">> Saved speaker embeddings to %s" %spkembed_arkscp)


def auto_gpus_select(maxGpuNum):
  from GPUtil import getAvailable
  from time import sleep
  while True:
    gpu_idxs = getAvailable(order='memory', limit=maxGpuNum, maxLoad=0.2, maxMemory=0.2, includeNan=False)
    if len(gpu_idxs):
      break
    else:
      print("No available GPU! Wait for other users to exit...")
      sleep(10)
  return gpu_idxs
  
