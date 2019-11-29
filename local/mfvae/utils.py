import pdb
import os
import sys
# import subprocess, io, threading
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
  

def prepare_device(str_cuda_ids):
  import torch
  n_gpus = torch.cuda.device_count()
  cuda_ids = set(int(gpu_idx) for gpu_idx in str_cuda_ids.split(
      ",") if gpu_idx.strip().isdigit())
  # Use CPUs.
  if n_gpus <= 0: 
    print("Warning: No GPU available in this device. Evaluation will be performed on CPU.")
    device = torch.device('cpu')
    return device, []
  # not specify gpu id-> try auto select a GPU
  if len(cuda_ids) == 0:
    cuda_ids = auto_gpus_select(1)
  # the smallest cuda id is regarded as root device.
  min_cuda_id = min(cuda_ids)
  assert min_cuda_id >= 0, "Invalid GPU index:{} .".format(
      min_cuda_id)
  for cuda_id in cuda_ids:
    assert cuda_id < n_gpus, "GPU(:{} is not available on this machine.".format(
        cuda_id)
  device = torch.device('cuda:{}'.format(min_cuda_id))
  print("Select GPU(s): {}".format(','.join([str(num) for num in list(cuda_ids)])))
  return device, list(cuda_ids)

