from __future__ import print_function, division
import os
import torch
import numpy as np
import kaldi_io
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from os.path import join
from utils import ConfigBase, read_gmm_model, Logger
import pdb
from kaldi.util.table import RandomAccessMatrixReader, MatrixWriter, SequentialMatrixReader, SequentialIntReader, RandomAccessVectorReader
from kaldi.transform.cmvn import Cmvn
# import errno
from random import randint
# from kaldi.feat._feature_functions import SlidingWindowCmnOptions, sliding_window_cmn

# This script load the fbanks for each utterance.
# In addition, this script adds data agumentation in the feature map as
# an option and attempts to boost its performance.

class Config(ConfigBase):
  def __init__(self):
    super(Config, self).__init__()
    self.dataloader_config = "======Data Preprocessing Configuration======"
    # self.num_spks = 1251
    self.feat_dim = 64
    self.frame_num_thresh = 4000
    self.data_crop = False
    self.sort_utts = False
    # self.cmvn_apply_global = True
    # self.cmvn_norm_vars = False
    # self.cmvn_reverse = False
    # self.cmvn_stats_rxfilename = ""
    self.gmvn_apply = True
    self.gmvn_norm_vars = False
    self.gmvn_stats_rxfilename = ""
    # self.vad_label = False
    # self.use_ivec = False
    # self.data_aug = False
    # self.spk_label = True
    # self.train_cln_data_dir = 'data/train'


def scp2dict(ipath2scp):
  fd = kaldi_io.open_or_fd(ipath2scp)  # ipath2scp can be a pipeline
  id2path = {}
  for line in fd:
    (id, path) = line.decode("utf-8").rstrip().split(' ', 1)
    id2path[id] = path
  return id2path


def pad_tensor(vec, pad, dim):
  """
  args:
      vec - tensor to pad
      pad - the size to pad to
      dim - dimension to pad

  return:
      a new tensor padded to 'pad' in dimension 'dim'
  """
  pad_size = list(vec.shape)
  pad_size[dim] = pad - vec.size(dim)
  return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate(object):
  """
  a variant of callate_fn that pads according to the longest sequence in
  a batch of sequences
  """

  def __init__(self, dim=0):
    """
    args:
        dim - the dimension to be padded (dimension of time in sequences)
    """
    self.dim = dim

  def pad_collate(self, batch):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
    """
    # find longest sequence
    max_len = max(map(lambda x: x[0].shape[self.dim], batch))
    xs = []
    ys = []
    for x, y in batch:
      xs.append(pad_tensor(x, pad=max_len, dim=self.dim))
      ys.append(y)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)

    return xs, ys

  def __call__(self, batch):
    return self.pad_collate(batch)


class CropCollate(object):
  """
  a variant of callate_fn that crops according to the shortest sequence in
  a batch of sequences
  """

  def __init__(self, dim=0):
    """
    args:
        dim - the dimension to be cropped (dimension of time in sequences)
    """
    self.dim = dim
    # self.dim = 0
    # self.crop_labels = label_is_vad

  def pad_collate(self, batch):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after cropping
        ys - a LongTensor of all labels in batch
    """
    # find shortest sequence
    min_len = min(map(lambda x: x[0].shape[self.dim], batch))
    xs = []
    ys = []
    for x, y in batch:
      xs.append(x[:min_len, :])
      # if self.crop_labels:
      #   ys.append(y[:min_len])
      # else:
      ys.append(y)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)

    return xs, ys

  def __call__(self, batch):
    return self.pad_collate(batch)


class MyDataset(Dataset):
  """
    Dataset for utterance-embedding and speaker labels.
  """

  def __init__(self, feats_scp, config):
    # load dicts
    uttids = list(scp2dict(feats_scp))
    num_utts = len(uttids)

    # # create labels
    # utt2spk = scp2dict(utt2spk_scp)
    # if config.spk_label:
    #   spks = utt2spk.values()
    #   ints = []
    #   # Only for voxceleb1
    #   for spk in spks:
    #     ints.append(int(spk[-4:])-1)
    #   utt2intspk = dict(zip(utt2spk.keys(), ints))
    #   self.utt2intspk = utt2intspk
    
    self.gmvn_apply = config.gmvn_apply
    self.gmvn_norm_vars = config.gmvn_norm_vars
    if self.gmvn_apply_global:
      gmvn = Cmvn()
      gmvn.read_stats(config.gmvn_stats_rxfilename)
      self.gmvn_normalizer = gmvn
    # self.spk_label = config.spk_label
    # self.vad_label = config.vad_label
    # self.vad_scp = vad_scp
    self.feats_scp = feats_scp
    # self.use_ivec = config.use_ivec
    # self.ivec_scp = ivec_scp
    self.sort_utts = config.sort_utts
    self.uttids = uttids 
    self.num_utts = num_utts
    self.feat_dim = config.feat_dim
    # self.num_spks = config.num_spks
    self.frame_num_thresh = config.frame_num_thresh
    self.data_crop = config.data_crop
    # self.data_aug = config.data_aug
    # self.cln_feats_scp = config.train_cln_data_dir + '/feats.scp'

    print("Loading dataset...")
    assert os.path.isfile(feats_scp), \
        "{} not found. Exit!".format(feats_scp)
    self.utt2feats = self.read_utt2feats(feats_scp)
    # if self.data_aug:
    #   assert os.path.isfile(self.cln_feats_scp), \
    #     "{} not found. Exit!".format(self.cln_feats_scp)
    #   self.utt2clnfeats = self.read_utt2feats(self.cln_feats_scp)
    # if self.vad_label:
    #   self.utt2vad = self.read_utt2vads(vad_scp)
    # assert False == (self.vad_label and self.use_ivec)
    # if self.use_ivec:
    #   self.utt2ivec = self.read_utt2ivec(ivec_scp)

  def __len__(self):
    return self.num_utts

  def __getitem__(self, index):
    uttid = self.uttids[index]
    # feats is np.array with shape (num_frame, feat_dim)
    feats = self.utt2feats[uttid]
    if self.gmvn_apply:
      self.global_normalizer.apply(feats, norm_vars=self.gmvn_norm_vars)
    feats = self.fix_feats_num(feats.numpy(), self.frame_num_thresh)
    if self.data_crop:
      feats = self.random_crop_frame(feats, num_frames_thresh=300)
    label = 0  # compatible to PadCollate/CropCollate
   return torch.FloatTensor(feats), torch.LongTensor([label]).squeeze_()
  
  def random_crop_frame(self, feats, num_frames_thresh=100):
    num_frame = feats.shape[0]
    assert num_frames_thresh < num_frame
    frame_start_idx = np.random.randint(num_frame-num_frames_thresh+1)
    return feats[frame_start_idx:frame_start_idx+num_frames_thresh]

  def read_utt2feats(self, ipath2feats):
    rspec = 'scp:{0}'.format(ipath2feats)
    utt2feats = RandomAccessMatrixReader(rspec)
    return utt2feats

  def fix_feats_num(self, feats, frame_num_thresh):
    frame_num = feats.shape[0]
    if frame_num >= frame_num_thresh:
      feats = feats[:frame_num_thresh, :]
    return feats

  def rand_sort_uttids(self, feats_scp):
    """ Sort uttids according to its duration added with noise
        Note that, this func is applied to deal with variable-length input for minibatch training.
        Utts in each batch share similar  durations.
        To add some randomness, a noise dur (0~100s) is involved in sorting.
    """
    # SequentialIntReader or RandomAccessIntReader
    with SequentialIntReader("ark:feat-to-len scp:{} ark:-|".format(feats_scp)) as utt2len:
      noise = 10000  # to add randomness within a minibatch
      list_utt_dur = [(utt, length+randint(0, noise))
                      for utt, length in utt2len]
      sorted_utt_dur = sorted(list_utt_dur, key=lambda kv: kv[1])
      sorted_utts = [utt for utt, _ in sorted_utt_dur]

    return sorted_utts

  def shuffle_uttids(self):
    self.uttids = self.rand_sort_uttids(self.feats_scp)


def get_dataset(data_dir, config):
  # ipath2label = "{0}/utt2spk".format(data_dir)
  feats_scp = "{0}/feats.scp".format(data_dir)
  # vad_scp = "{0}/vad.scp".format(data_dir)
  # ivec_scp = "{0}/ivector.scp".format(data_dir)
  dataset = MyDataset(feats_scp, config)
  return dataset


if __name__ == '__main__':
  config = Config()
  config.parse_args()  # revise configurations from cmd line
  config.print_args()
  dataset = get_dataset('data/train', config)
  # debug
  print("Pass test!")
