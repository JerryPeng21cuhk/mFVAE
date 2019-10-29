from model import mFVAE
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import os
import sys
import time
import datetime
from solver import SolverBase
import pdb
from torch.utils.data import DataLoader
from dataloader import get_dataset, CropCollate
from dataloader import Config as DataLoaderConfig
from sampler import RandomBatchSampler
from model import Config as ModelConfig
from solver import Config as SolverConfig
from utils import auto_gpus_select


class Config(DataLoaderConfig, ModelConfig, SolverConfig):
  def __init__(self):
    super(Config, self).__init__()
    self.train_config = "======Training Configuration====="

    ## override DataLoaderConfig
    self.feat_dim = 64
    self.frame_num_thresh = 4000
    self.data_crop = True
    self.sort_utts = True
    self.gmvn_apply = True
    self.gmvn_norm_vars = True
    self.gmvn_stats_rxfilename = "data/train/cmvn_stats"

    ## override SolverConfig
    self.log_dir = 'exp/unsup_vae/log'
    self.model_save_dir = 'exp/unsup_vae'
    self.gpu_idxs = ""
    self.batch_size = 64
    self.num_epochs = 50
    self.resume_epoch = 0
    self.log_step = 150
    self.init_lr = 1e-3
    self.end_lr = 1e-4

    ## override ModelConfig
    self.cluster_num = 128
    self.embed_dim = 600
    self.bnf_feat_dim = 64

    ## new config
    self.num_workers = 10
    self.train_data_dir = 'data/train'
    self.log_filename = 'unsup_train.log'
    self.adam_beta1 = 0.9
    self.adam_beta2 = 0.999


class Solver(SolverBase):
  def __init__(self, train_loader, config):
    SolverBase.__init__(self, train_loader, config)
    # Model Configuration
    # self.num_spks = config.num_spks
    self.feat_dim = config.feat_dim
    # self.frame_num_thresh = config.frame_num_thresh
    self.cluster_num = config.cluster_num
    self.embed_dim = config.embed_dim
    self.bnf_feat_dim = config.bnf_feat_dim
    # self.latent_dim = config.latent_dim
    # self.clust_downsample = config.clust_downsample
    # self.feat_downsample = config.feat_downsample
    # self.left_context = config.left_context
    # self.right_context = config.right_context

    self.adam_beta1 = config.adam_beta1
    self.adam_beta2 = config.adam_beta2
    self.sort_utts = config.sort_utts

    self.model = mFVAE(D=self.feat_dim,
                       K=self.cluster_num,
                       bnf_feat_dim=self.bnf_feat_dim,
                       embed_dim=self.embed_dim)

    # load model
    if self.resume_epoch > 0:
      assert self.resume_epoch < self.num_epochs, "resume_epoch {0} should be less than num_epochs {1}".format(
          self.resume_epoch, self.num_epochs)
      print("resuming epoch {} ...".format(self.resume_epoch))
      self.restore_model(self.resume_epoch)
    else:
      assert self.resume_epoch == 0, "Invalid config: resume_epoch {}".format(
          self.resume_epoch)
    # send model to GPUs/CPU (2 GPUs at most)
    if not config.gpu_idxs:
      config.gpu_idxs = ','.join([str(num) for num in auto_gpus_select(1)]) # auto mode
    self.root_device = self.send_model2device(config.gpu_idxs)
    # optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters(), self.init_lr, [
                                      self.adam_beta1, self.adam_beta2]) #, weight_decay=1e-5)
    self.print_network(self.model, 'mixture Factorized VAE')

  def send_model2device(self, str_cuda_idxs):
    device, device_ids = self.prepare_device(str_cuda_idxs)
    if len(device_ids) > 1:
      self.multi_gpu = True
      self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
    self.model = self.model.to(device)
    return device

  def restore_model(self, resume_epoch):
    """Restore model"""
    print('Loading the trained model from step {}...'.format(resume_epoch))
    path2model = os.path.join(self.model_save_dir, '{}.mdl'.format(resume_epoch))
    checkpoint = torch.load(path2model)
    self.model.load_state_dict(checkpoint['model'])

  def save_model(self, epoch):
    path2model = os.path.join(self.model_save_dir, '{}.mdl'.format(epoch))
    if self.multi_gpu:
      torch.save({
          'model': self.model.module.state_dict(),
          }, path2model)
    else:
      torch.save({
          'model': self.model.state_dict(),
          }, path2model)
    print('Saved model checkpoints into {}...'.format(path2model))

  def train(self):
    train_loader = self.dataloader
    # lr
    init_lr = self.init_lr  # 1e-4
    end_lr = self.end_lr  # 1e-5
    lrs = self.exp_decayed_lr(init_lr, end_lr, self.num_epochs)
    print(lrs)

    # start training
    print('Start training...')
    start_time = time.time()
    self.model.train()  # set to training mode
    for epoch in range(self.resume_epoch, self.num_epochs, 1):
      lr = lrs[epoch]  # update lr
      self.update_lr(lr)
      print('Decayed learning rates, lr: {}.'.format(lr))
      if self.sort_utts:
        train_loader.dataset.shuffle_uttids()
      for i, (feat2d, _) in enumerate(train_loader):
        feat2d = feat2d.to(self.root_device)
        # with torch.autograd.set_detect_anomaly(True):
        label_loss, reg_qy, reg_qw, qy_ideal = self.model(feat2d)
        train_loss = label_loss # + reg_qw * 0.1
        # train_loss = label_loss + qy_ideal + 0.1 * discrim_loss
        # backward
        self.reset_grad()
        train_loss.backward()
        self.optimizer.step()
        # Print out training information
        if (i+1) % self.log_step == 0:
          et = time.time() - start_time
          et = str(datetime.timedelta(seconds=et))[:-7]
          log = "Elapsed [{}], Iteration [{}/{}], Epoch [{}/{}]".format(
              et, i+1, self.num_iters, epoch+1, self.num_epochs)
          log += ", {}: {:.4f}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}".format("training loss", train_loss.item(),
                                                   "label_loss", label_loss.item(),
                                                   "reg_qy", reg_qy.item(),
                                                   "reg_qw", reg_qw.item(),
                                                   "qy_ideal", qy_ideal.item())
          print(log)
        # del train_loss, entropy
      self.save_model(epoch+1)
      # if True == self.earlystopping.stop(valid_loss_acc):
      #   break


def main(config):
  trainset = get_dataset(config.train_data_dir, config)
  blockSampler = RandomBatchSampler(trainset, config.batch_size, drop_last=True)
  train_loader = DataLoader(dataset=trainset, batch_sampler=blockSampler,
                            collate_fn=CropCollate(0), num_workers=config.num_workers)
  solver = Solver(train_loader, config)
  solver.train()


if __name__ == '__main__':
  from utils import Logger
  config = Config()
  config.parse_args()  # revise configurations from cmd line
  # Create directories if not exist
  if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)
  if not os.path.exists(config.model_save_dir):
    os.makedirs(config.model_save_dir)
  # Results will be printed on screen and log file
  sys.stdout = Logger('{0}/{1}'.format(config.log_dir, config.log_filename))
  config.print_args()

  main(config)
