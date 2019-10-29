# Copyright        2018 Jerry Peng
# Apache 2.0

# This script provides a template to train NN models. It will be further used
# in train_mfvae.py script.

from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import os
import sys
import time
import datetime
import pdb
from torch.utils.data import DataLoader


class Config(object):
  def __init__(self):
    super(Config, self).__init__()
    self.solver_config = "======Solver Configuration======"
    self.gpu_idxs = ""
    self.log_step = 150
    self.log_dir = ""
    self.batch_size = 16
    self.resume_epoch = 0
    self.num_epochs = 10
    self.init_lr = 1e-3
    self.end_lr = 1e-4
    self.model_save_dir = ""


class SolverBase(object):
  def __init__(self, dataloader, config):
    self.dataloader = dataloader
    self.batch_size = config.batch_size
    self.num_epochs = config.num_epochs
    self.num_iters = len(dataloader)
    self.resume_epoch = config.resume_epoch
    self.init_lr = config.init_lr
    self.end_lr = config.end_lr
    self.log_dir = config.log_dir
    self.model_save_dir = config.model_save_dir
    self.log_step = config.log_step
    self.loss_criterion = None
    self.multi_gpu = False

  def send_model2device(self, str_cuda_idxs):
    device, device_ids = self.prepare_device(str_cuda_idxs)
    if len(device_ids) > 1:
      self.multi_gpu = True
      self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
    self.model = self.model.to(device)
    return device

  def prepare_device(self, str_cuda_ids):
    n_gpus = torch.cuda.device_count()
    cuda_ids = set(int(gpu_idx) for gpu_idx in str_cuda_ids.split(",") if gpu_idx.strip().isdigit())
    # Use CPUs.
    if n_gpus <= 0 or len(cuda_ids) == 0:
      if len(cuda_ids) > 0:
        print("Warning: No GPU available in this device. Training iwll be performed on CPU.")
      device = torch.device('cpu')
      return device, []
    # Use GPUs.
    min_cuda_id = min(cuda_ids)
    assert min_cuda_id >= 0, "Invalid GPU index:{} available on this machine.".format(min_cuda_id)
    # pdb.set_trace()
    for cuda_id in cuda_ids:
      assert cuda_id < n_gpus, "There is no GPU:{} available on this machine.".format(cuda_id)
    device = torch.device('cuda:{}'.format(min_cuda_id))
    return device, list(cuda_ids)

  def print_network(self, model, name):
    """Print out the network information"""
    num_params = 0
    for p in model.parameters():
      num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

  def restore_model(self, resume_epoch):
    """Restore model"""
    print('Loading the trained model from step {}...'.format(resume_epoch))
    path2model = os.path.join(self.model_save_dir, '{}.mdl'.format(resume_epoch))
    self.model.load_state_dict(torch.load(path2model, map_location=lambda storage, loc: storage))

  def save_model(self, epoch):
    path2model = os.path.join(self.model_save_dir, '{}.mdl'.format(epoch))
    if self.multi_gpu:
      torch.save(self.model.module.state_dict(), path2model)
    else:
      torch.save(self.model.state_dict(), path2model)
    print('Saved model checkpoints into {}...'.format(path2model))

  def update_lr(self, lr):
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

  def reset_grad(self):
    self.optimizer.zero_grad()

  def lin_decayed_lr(self, init_lr, end_lr, num_epochs):
    lrs = np.linspace(init_lr, end_lr, num=self.num_epochs)
    return lrs

  def exp_decayed_lr(self, init_lr, end_lr, num_epochs):
    scalar = (end_lr / init_lr) ** (1.0 / (num_epochs-1))
    lrs = [0] * num_epochs
    lr = init_lr
    lrs[0] = lr
    for i in range(num_epochs-1):
      lr = scalar * lr
      lrs[i+1] = lr
    return lrs

  def validation_loss(self, valid_loader):
    valid_loss_acc = 0.0
    correct = 0
    total = 0
    self.model.eval()  # set to eval mode
    for i, (feat2d, label) in enumerate(valid_loader):
      feat2d.unsqueeze_(1)
      feat2d = feat2d.to(self.root_device)
      label = label.to(self.root_device)

      # forward
      pred = self.model(feat2d)
      _, predicted = torch.max(pred.data, 1)
      total += label.size(0)
      correct += (predicted == label).sum().item()
      valid_loss = self.loss_criterion(pred, label)
      # pdb.set_trace()
      valid_loss_acc += valid_loss.item()
    self.model.train()  # set to train mode
    return valid_loss_acc, 100 * correct / total

  def train(self):
    pass
