import torch
import torch.nn as nn
import pdb
from tdnn import TDNN
# from tdnn_with_embed import TDNN_with_embed
import math
import torch.nn.functional as F
# from torch.nn.modules.utils import _pair, _quadruple
# from torch.nn.parameter import Parameter
import numpy as np
# from dataloader import Config as DataLoaderConfig
from utils import ConfigBase
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


class Config(ConfigBase):
  def __init__(self):
    super(Config, self).__init__()
    self.model_config = "======Model Configuration======"
    self.cluster_num = 128  # K
    self.embed_dim = 600
    self.bnf_feat_dim = 64


class SwapDim(nn.Module):
  """ Swap two dimensions. Used to convert C dim to H or W.

  Args:
                   dim1: target dim1
                   dim2: target dim2
  """

  def __init__(self, dim1, dim2):
    super(SwapDim, self).__init__()
    self.dim1 = dim1
    self.dim2 = dim2

  def forward(self, x):
    x = x.transpose(self.dim1, self.dim2)
    return x


# HLoss_logit is the summation over a batch.
class HLoss_logit(nn.Module):
  def __init__(self):
    super(HLoss_logit, self).__init__()

  def forward(self, x):
    """x is assumed to be logits p(x)"""
    # scalar = torch.prod(torch.tensor(x.shape[:-1]).float())
    result = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
    result = -1.0 * result.sum()
    return result


class FrameCompressor(nn.Module):
  def __init__(self, feat_dim, output_dim):
    super(FrameCompressor, self).__init__()
    self.frame1 = TDNN(input_dim=feat_dim, output_dim=512, context_size=5, dilation=1)
    self.frame2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2)
    self.frame3 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3)
    self.frame4 = TDNN(input_dim=512, output_dim=output_dim, context_size=1, dilation=1)

  def forward(self, x):
    x = self.frame1(x)
    x = self.frame2(x)
    x = self.frame3(x)
    x = self.frame4(x)
    return x


class QY(nn.Module):
  """ q(y|x) """
  def __init__(self, D, K, temper=0.1):
    super(QY, self).__init__()
    self.fc1 = nn.Sequential(
      nn.Linear(D, 512),
      SwapDim(1, 2),
      nn.BatchNorm1d(512),
      SwapDim(1, 2),
      nn.ReLU(),
    )
    self.fc2 = nn.Sequential(
      nn.Linear(512, 512),
      SwapDim(1, 2),
      nn.BatchNorm1d(512),
      SwapDim(1, 2),
      nn.ReLU()
    )
    self.fc3 = nn.Linear(512, K)
    self.temper = temper

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    logits = self.fc3(x)
    B,L,K = logits.shape
    RelaxedOneHotSampler = RelaxedOneHotCategorical(
                                  float(self.temper),
                                  logits=logits
                                  )
    y = RelaxedOneHotSampler.rsample()
    return y, F.softmax(logits, dim=-1), logits


class QW_xvec(nn.Module):
  """ q(wi|xit) Given obs xit predict wi [Gaussian assumed]"""
  def __init__(self, D, embed_dim):
    super(QW_xvec, self).__init__()
    self.featCompressor = FrameCompressor(feat_dim=D, output_dim=1500)
    self.fc1 = nn.Sequential(
      nn.BatchNorm1d(1500*2),
      nn.Linear(1500*2, embed_dim),
      nn.BatchNorm1d(embed_dim),
      nn.ReLU(),
    )
    self.qw_mean = nn.Linear(embed_dim, embed_dim)
    self.qw_var = nn.Sequential(
      nn.Linear(embed_dim, embed_dim),
      nn.Softplus()
    )

  def forward(self, x):
    bnf_x = self.featCompressor(x)
    mean_x = torch.mean(bnf_x, dim=1)
    var_x = torch.mean(bnf_x**2, dim=1) - mean_x**2
    # mean_x = F.normalize(mean_x, p=2, dim=1)
    # var_x = F.normalize(var_x, p=2, dim=1)
    pool_x = torch.cat((mean_x, var_x), dim=-1)
    x = self.fc1(pool_x)
    mean = self.qw_mean(x)
    var = self.qw_var(x)
    eps = torch.randn_like(mean)
    return mean + eps * torch.sqrt(var), mean, var


class PX(nn.Module):
  """ p(xit|yit, wi), Given Tcwi+mu, predict xit [L2 norm only mean] """
  def __init__(self, K, embed_dim, d, D):
    super(PX, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(embed_dim+3*K, 512),
        SwapDim(1,2),
        nn.BatchNorm1d(512),
        SwapDim(1,2),
        nn.ReLU()
      )
    self.layer2 = nn.Sequential(
        nn.Linear(512+embed_dim, 512),
        SwapDim(1,2),
        nn.BatchNorm1d(512),
        SwapDim(1,2),
        nn.ReLU()
      )
    self.layer3 = nn.Sequential(
        nn.Linear(512+embed_dim, 512),
        SwapDim(1,2),
        nn.BatchNorm1d(512),
        SwapDim(1,2),
        nn.ReLU()
      )
    self.px_mean = nn.Sequential(
      nn.Linear(512+embed_dim, 512),
      SwapDim(1,2),
      nn.BatchNorm1d(512),
      SwapDim(1,2),
      nn.ReLU(),
      nn.Linear(512, D)
    )

  def forward(self, x):
    y, w = x
    y = y.unsqueeze(1)
    y = F.unfold(
      y,
      (3, y.shape[-1]),
      stride=(1, y.shape[-1]),
      dilation=(1, 1)
    )
    y = y.transpose(1,2)  # (B, L, K)
    w = w.unsqueeze(1)
    wshape = list(y.shape)
    wshape[-1] = w.shape[-1]
    x = self.layer1(torch.cat((y, w.expand(wshape)), -1))
    x = self.layer2(torch.cat((x, w.expand(wshape)), -1))
    x = self.layer3(torch.cat((x, w.expand(wshape)), -1))
    return self.px_mean(torch.cat((x, w.expand(wshape)), -1))


class mFVAE(nn.Module):
  """ Inspired by i-vector method
      Here we also apply two reparameterization tricks on w and y.
      In the paper, we only mentioned the trick on y and put this simplification in section:
        from mFVAE to mFAE
  """
  def __init__(self, D, K, bnf_feat_dim, embed_dim):
    super(mFVAE, self).__init__()
    self.featCompressor = FrameCompressor(feat_dim=D, output_dim=bnf_feat_dim)
    self.qy = QY(bnf_feat_dim, K)
    self.qw_xvec = QW_xvec(D, embed_dim=embed_dim)
    self.px = PX(K, embed_dim=embed_dim, d=d, D=D)
    self.K = K
    self.D = D
    self.embed_dim = embed_dim
    self.entropy = HLoss_logit()

  def forward(self, x):
    B, L, D = x.shape
    assert L > 14
    L = L - 14
    w, qw_mean, qw_var = self.qw_xvec(x)
    bnf_x = self.featCompressor(x)
    lab_x = x[:,8:-8,:]
    y, qy, qy_logits = self.qy(bnf_x)
    px_mean = self.px((y, w))
    ent = self.entropy(qy_logits)
    loss_avg_qy = math.log(self.K) - self.entropy(torch.sum(qy_logits, dim=(0,1))/(B*L))
    # loss_avg_qy is not mentioned in our paper. It is used to verify an argument in section 2.2.1
    loss_label = torch.sum((lab_x - px_mean)**2)
    loss_label = 0.5 * loss_label / (B*L) + self.D / 2.0 * math.log(2*math.pi)
    loss_reg_qw = torch.sum(qw_var) + torch.sum(qw_mean**2) - torch.log(qw_var).sum()
    loss_reg_qw = 0.5 * loss_reg_qw / (B*L) - 0.5 * self.embed_dim /L
    loss_reg_qy = -ent/(B*L) + math.log(self.K)
    return loss_label, loss_reg_qy, loss_reg_qw, loss_avg_qy

  def get_feat(self, x):
    x, vec = x
    B = x.shape[0]
    bnf_x = self.featCompressor(x)
    _, qy, _ = self.qy(bnf_x)
    # w, qw_mean, qw_var = self.qw_xvec(x)
    non_info_embed = vec.unsqueeze(0).expand((B, -1))
    px_mean = self.px((qy, non_info_embed))
    return px_mean

  def get_post(self, x):
    bnf_x = self.featCompressor(x)
    _, qy, _ = self.qy(bnf_x)
    return qy

  def get_embed(self, x):
    w, qw_mean, qw_var = self.qw_xvec(x)
    return qw_mean


class mFAE(mFVAE):
  """ Inspired by i-vector method
  """
  def __init__(self, D, K, bnf_feat_dim, embed_dim):
    super(mFAE, self).__init__()

  def forward(self, x):
    B, L, D = x.shape
    assert L > 16
    L = L - 16
    w, qw_mean, qw_var = self.qw_xvec(x)
    bnf_x = self.featCompressor(x)
    lab_x = x[:,8:-8,:]
    y, qy, qy_logits = self.qy(bnf_x)
    px_mean = self.px((y, qw_mean))
    loss_label = torch.sum((lab_x - px_mean)**2)
    loss_label = 0.5 * loss_label / (B*L) + self.D / 2.0 * math.log(2*math.pi)
    return loss_label

