"""TODO(taylanbil): DO NOT SUBMIT without one-line documentation for test_train_fairseq.

TODO(taylanbil): DO NOT SUBMIT without a detailed description of test_train_fairseq.


  model_parallel = dp.DataParallel(torchvision_model, device_ids=devices)
    tracker = xm.RateTracker()
      xm.optimizer_step(optimizer)

    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)


    tracker = xm.RateTracker()
      tracker.add(FLAGS.batch_size)
      print(tracker.rate()) gibi bisey

      print(torch_xla._XLAC._xla_metrics_report())
"""


import argparse
import sys
import os

pytorch_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
xla_folder = os.path.join(pytorch_folder, 'xla')
fairseq_folder = os.path.join(pytorch_folder, 'fairseq')
sys.path.insert(0, xla_folder)
sys.path.insert(0, fairseq_folder)
sys.path.insert(0, pytorch_folder)

import torch
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm

import train as fairseq_train
from fairseq.trainer import Trainer as FairseqTrainer
from fairseq import options

raise
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--num_cores', type=int, default=8)
parser.add_argument('--log_steps', type=int, default=1)
FLAGS = parser.parse_known_args()
DEVICES = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)


class Trainer(FairseqTrainer):

  WHOAMI = 'xla :)'

  def __init__(self, **kwargs):
    super(Trainer, self).__init__(**kwargs)

  #def train_step(self, samples, dummy_batch=False, raise_oom=False):
  def train_step(self, samples):
    pass


if __name__ == '__main__':
  parser = options.get_training_parser()
  args = options.parse_args_and_arch(parser)
  # override certain args so that we use XLA parallelism instead of torch.
  args.distributed_init_method = None
  print(args)
  import pdb
  pdb.set_trace()
