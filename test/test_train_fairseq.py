"""TODO(taylanbil): DO NOT SUBMIT without one-line documentation for test_train_fairseq.

TODO(taylanbil): DO NOT SUBMIT without a detailed description of test_train_fairseq.
"""


import argparse
import sys
import os
import math

pytorch_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
xla_folder = os.path.join(pytorch_folder, 'xla')
fairseq_folder = os.path.join(pytorch_folder, 'fairseq')
sys.path.insert(0, xla_folder)
sys.path.insert(0, fairseq_folder)

import torch

import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm

from fairseq.data import data_utils
# Overwriting collate_tokens to guarantee constant size input tensors
# This is reducing the number of graph recompiles
collate_tokens_generic = data_utils.collate_tokens


def collate_tokens_new(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """
    Copied over from fairseq.data_utils, and modified so that
    num_columns in the output tensor is not too variable.
    """
    # correcting columns
    global PAD_TO_LENGTH
    size = max(v.size(0) for v in values)
    if size > PAD_TO_LENGTH:
        print('I had to change PAD_TO_LENGTH from {} to {}, this is going to trigger graph recompiles'.format(PAD_TO_LENGTH, size))
        PAD_TO_LENGTH = size
    size = PAD_TO_LENGTH
    # done correcting
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


data_utils.collate_tokens = collate_tokens_new

from fairseq import options, tasks, checkpoint_utils
from fairseq.trainer import Trainer
from fairseq.data import iterators
from fairseq.meters import StopwatchMeter


def parse_args():
  # We need to control certain flags here.
  # e.g. parallelization needs to be suppressed and deferred to torch_xla flags
  # e.g. input tensor shapes need to be controlled via
  #   max_sentences, required_batch_size_multiple
  parser = options.get_training_parser()
  parser.add_argument('--num_cores', type=int, default=8)
  parser.add_argument('--pad_to_length', type=int, default=64)
  parser.add_argument('--use_gpu', action='store_true')
  FLAGS = options.parse_args_and_arch(parser)
  if not FLAGS.use_gpu:
    if FLAGS.fp16:
      print('suppressing "fp16"')
      FLAGS.fp16 = False
    if FLAGS.distributed_world_size > 1:
      print('suppressing "distributed_world_size"')
      FLAGS.distributed_world_size = 1
    if FLAGS.distributed_init_method is not None:
      print('suppressing "distributed_init_method"')
      FLAGS.distributed_init_method = None
    if FLAGS.max_sentences != FLAGS.required_batch_size_multiple:
      batch_size = max(filter(lambda r: r is not None, [FLAGS.max_sentences, FLAGS.required_batch_size_multiple]))
      print(
        '"max_sentences" and "required_batch_size_multiple" must be equal'
        ' to have good performance on TPUs. Using {}'.format(batch_size))
      FLAGS.max_sentences = batch_size
      FLAGS.required_batch_size_multiple = batch_size
    if FLAGS.max_tokens is not None:
      print('"max_tokens" needs to be None for better TPU performance')
      FLAGS.max_tokens = None
  return FLAGS


def prepare_task(args):
  # Setup task, e.g., translation, language modeling, etc.
  task = tasks.setup_task(args)

  # Load valid dataset (we load training data below, based on the latest checkpoint)
  for valid_sub_split in args.valid_subset.split(','):
    task.load_dataset(valid_sub_split, combine=True, epoch=0)

  # Build models and criteria to print some metadata
  criterion = task.build_criterion(args)
  model_parallel = dp.DataParallel(
    lambda: task.build_model(args), device_ids=DEVICES, drop_last=True)
  criteria = {device: task.build_criterion(args)
              for device in model_parallel._device_ids}
  models = {model_parallel._get_model_device(model): model
          for model in model_parallel._models}
  model, criterion = model_parallel._models[0], list(criteria.values())[0]
  print(model)
  print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
  print('| num. model params: {} (num. trained: {})'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() for p in model.parameters() if p.requires_grad),
  ))
  del model, criterion

  # Build trainers
  trainers = {device: Trainer(args, task, models[device], criteria[device])
              for device in model_parallel._device_ids}
  trainer = trainers[DEVICES[0]]
  lr = trainer.get_lr()

  # TODO(taylanbil): for now, this next line is only creating the iterator.
  # validate its behavior with the case where a checkpoint actually exists.

  # Load the latest checkpoint if one is available and restore the
  # corresponding train iterator
  extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
  return task, trainers, model_parallel, epoch_itr, lr


def main_tpu(args):

  def train_loop_fn(model, loader, device, context):
    trainer = trainers[str(device)]
    tracker = xm.RateTracker()
    for i, samples in loader:
      print('device {}, step {}: begin'.format(device, i))
      trainer.train_step(samples)
      xm.optimizer_step(trainer.optimizer)
      print('device {}, step {}: end'.format(device, i))
      tracker.add(BATCH_SIZE)
    return tracker

  def valid_loop_fn(model, loader, device, context):
    raise

  def keep_training(lr, epoch_itr, trainers):
    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = min(trainer.get_lr() for trainer in trainers.values())
    n_updates = max(trainer.get_num_updates() for trainer in trainers.values())
    return (lr > FLAGS.min_lr) and (epoch_itr.epoch < max_epoch) and (n_updates < max_update)

  print("Args\n---------")
  print(args)

  task, trainers, model_parallel, epoch_itr, lr = prepare_task(args)

  train_meter = StopwatchMeter()
  train_meter.start()
  valid_losses = [None]
  valid_subsets = args.valid_subset.split(',')
  while keep_training(lr, epoch_itr, trainers):
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    train_loader = iterators.GroupedIterator(itr, update_freq)
    trackers = model_parallel(train_loop_fn, train_loader)
    print('Tracker Rates:')
    for tracker in trackers:
        print('\tRate={:.2f}'.format(tracker.rate()))
    print(torch_xla._XLAC._xla_metrics_report())

    # if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
    #   # TODO(taylanbil): implement validate
    #   valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
    # else:
    #   valid_losses = [None]
    #
    # # TODO(taylanbil): verify the learning rate update
    # from fairseq import pdb
    # pdb.set_trace()
    # # only use first validation loss to update the learning rate
    # lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
    # pdb.set_trace()
    #
    # # save checkpoint
    # if epoch_itr.epoch % args.save_interval == 0:
    #   checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

  train_meter.stop()
  print('| done training in {:.1f} seconds'.format(train_meter.sum))


if __name__ == '__main__':
  # override certain args so that we use XLA parallelism instead of torch.
  FLAGS = parse_args()
  if FLAGS.use_gpu:
    import train as fairseq_train
    fairseq_train.cli_main()
  else:
    BATCH_SIZE = FLAGS.max_sentences
    DEVICES = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)
    PAD_TO_LENGTH = FLAGS.pad_to_length
    main_tpu(FLAGS)

