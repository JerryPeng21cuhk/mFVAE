import torch
from math import ceil
from torch.utils.data.sampler import Sampler


class RandomBatchSampler(Sampler):
  """Sample blocks(batches) of sorted elments randomly. If without replacement, then sample from a blockwise shuffled dataset.
    If with replacement, then user can specify ``num_blocks`` to draw.

    Arguments:
      data_source (Dataset): dataset to sample from.
                             The dataset is assumed to be sorted according to some sample properties, say, length. (In my case, each sample is an utterance. Length means the utterance duration)
                             For the reason, pls refer to fastdataloader_sort.py.
      block_size (int): batch size.
      replacement (bool): blocks are drawn with replacement if ``True``, default=``False``
      num_blocks (int): numer of blocks to draw, default=`len(dataset)/len(block)`. 
                        This argument is supposed to be specified only when `replacement` is ``True``.
  """

  def __init__(self, data_source, block_size, drop_last=False, replacement=False, num_blocks=None):
    self.data_source = data_source
    self.replacement = replacement
    self.drop_last = drop_last
    self.block_size = block_size

    self._num_blocks = num_blocks  # put it after the above init
    if not isinstance(self.replacement, bool):
      raise ValueError("replacement should be a boolean value, but got "
                       "replacement={}".format(self.replacement))

    if not isinstance(self.block_size, int) or self.block_size <= 0:
      raise ValueError("block_size should be a postive integer "
                       "value, but got block_size={}".format(self.block_size))

    if self._num_blocks is not None and not replacement:
      raise ValueError("With replacement=False, num_blocks should not be specified, "
                       "since a random permute will be performed.")
    if not isinstance(self.num_blocks, int) or self.num_blocks <= 0:
      raise ValueError("num_blocks should be a positive integer "
                       "value, but got num_blocks={}".format(self.num_blocks))

    assert self.block_size <= len(data_source), "block_size > dataset size ?"

  @property
  def num_samples(self):
    return len(self.data_source)

  @property
  def num_blocks(self):
    if self._num_blocks is None:
      if self.drop_last:
        return self.num_samples // self.block_size
      else:
        return (self.num_samples + self.block_size - 1) // self.block_size
    else:
      return self._num_blocks

  def __iter__(self):
    if self.replacement:
      start_idxs = torch.randint(low=0, high=(
          self.num_samples-self.block_size+1), size=(self.num_blocks,), dtype=torch.int64).tolist()
      for start_idx in start_idxs:
        batch = [idx for idx in range(start_idx, start_idx+self.block_size)]
        yield batch
    else:
      start_idxs = torch.randperm(self.num_blocks).tolist()
      for start_idx in start_idxs:
        batch = [idx for idx in range(
            start_idx*self.block_size, min(start_idx*self.block_size+self.block_size, self.num_samples))]
        yield batch

  def __len__(self):
    return self.num_blocks
