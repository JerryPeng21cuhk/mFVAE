from model import mFVAE
import torch.nn as nn
import torch
import numpy as np
import os
import sys # import re
import time
import datetime
import pdb
from dataloader import get_dataset
from dataloader import Config as DataLoaderConfig
from model import Config as ModelConfig
from torch.utils.data import DataLoader
from utils import auto_gpus_select, prepare_device
from kaldi.util.table import PosteriorWriter, MatrixWriter


class Config(DataLoaderConfig, ModelConfig):
  def __init__(self):
    super(Config, self).__init__()
    self.posterior_config = "======Get Posterior Configuration======"
    ## Override DataLoaderConfig
    self.data_crop = False
    self.sort_utts = False
    self.gmvn_apply = True
    self.gmvn_norm_vars = True
    self.gmvn_stats_rxfilename = "data/train/cmvn_stats"

    ## new config
    self.batch_size = 1
    self.num_workers = 2
    self.gpu_idxs = "" # currently only single gpu is supported
    self.log_filename = 'compute_posts.log'
    self.log_dir = 'exp/unsup_mdl/log'
    self.ipath2model = 'exp/unsup_mdl/10.mdl'
    self.data_dir = 'data/train'
    self.post_dir = 'exp/train_posts'
    self.gpu_idxs = ""


class Evaluation(object):

  def __init__(self, config):
    self.batch_size = config.batch_size
    self.log_dir = config.log_dir
    # Model Configuration
    self.feat_dim = config.feat_dim
    self.cluster_num = config.cluster_num
    self.embed_dim = config.embed_dim
    # self.bnf_feat_dim = config.bnf_feat_dim

    self.ipath2model = config.ipath2model
    self.multi_gpu = False

    self.model = mFVAE(D=self.feat_dim, 
                       K=self.cluster_num,
                       embed_dim=self.embed_dim)
    self.load_model(self.ipath2model)
    # if not config.gpu_idxs:
    #   config.gpu_idxs = ','.join([str(num) for num in auto_gpus_select(1)]) # auto mode
    # print("select GPU device: {}".format(config.gpu_idxs))

    # send model to GPUs/CPU (1 GPU at most)
    self.root_device = self.send_model2device(config.gpu_idxs)
    self.print_network(self.model, 'mixture Factorized VAE')

  def print_network(self, model, name):
    """Print out the network information"""
    num_params = 0
    for p in model.parameters():
      num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

  def send_model2device(self, str_cuda_idxs):
    device, device_ids = prepare_device(str_cuda_idxs)
    assert len(device_ids) <= 1, \
        "multi-gpu mode for computing embeddings is not supported now"
    self.model = self.model.to(device)
    return device

  # def prepare_device(self, str_cuda_ids):
  #   n_gpus = torch.cuda.device_count()
  #   cuda_ids = set(int(gpu_idx) for gpu_idx in str_cuda_ids.split(
  #       ",") if gpu_idx.strip().isdigit())
  #   # Use CPUs.
  #   if n_gpus <= 0 or len(cuda_ids) == 0:
  #     if len(cuda_ids) > 0:
  #       print(
  #           "Warning: No GPU available in this device. Evaluation will be performed on CPU.")
  #     device = torch.device('cpu')
  #     return device, []
  #   # Use GPUs.
  #   min_cuda_id = min(cuda_ids)
  #   assert min_cuda_id >= 0, "Invalid GPU index:{} available on this machine.".format(
  #       min_cuda_id)
  #   for cuda_id in cuda_ids:
  #     assert cuda_id < n_gpus, "There is no GPU:{} available on this machine.".format(
  #         cuda_id)
  #   device = torch.device('cuda:{}'.format(min_cuda_id))
  #   return device, list(cuda_ids)

  def load_model(self, ipath2model):
    """Load model
    """
    print("Loading the model from {}".format(ipath2model))
    checkpoint = torch.load(ipath2model, map_location='cpu')
    self.model.load_state_dict(checkpoint['model'])

  def extract_post_to_hardisk(self, test_loader, post_wspecifier):
    print('>> Extracting utterance posteriors and write it to {}...'.format(post_wspecifier))
    uttids = test_loader.dataset.uttids
    # with PosteriorWriter(post_wspecifier) as post_writer:
    self.model.eval()
    with torch.no_grad():
      with MatrixWriter(post_wspecifier) as post_writer:
        for i, (feat2d, _) in enumerate(test_loader):
          feat2d = feat2d.to(self.root_device)
          post = self.model.get_post(feat2d)
          post_writer[uttids[i]] = post.squeeze().cpu().data.numpy()
    print('>> finish extracting utterance posteriors')


def main(config):
  dataset = get_dataset(config.data_dir, config)
  dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, drop_last=False)
  evaluator = Evaluation(config)
  if not os.path.exists(config.post_dir):
    os.makedirs(config.post_dir)
  evaluator.extract_post_to_hardisk(dataloader, 
                'ark,scp:{0}/posts.ark,{0}/posts.scp'.format(config.post_dir))


if __name__ == '__main__':
  from utils import Logger
  config = Config()
  config.parse_args()  # revise configurations from cmd line
  # Create directories if not exist
  if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)
  # Results will be printed on screen and log file
  sys.stdout = Logger('{0}/{1}'.format(config.log_dir, config.log_filename))
  config.print_args()  # show the revised configurations

  main(config)
