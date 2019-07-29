"""TODO(taylanbil): DO NOT SUBMIT without one-line documentation for test_train_fairseq.

TODO(taylanbil): DO NOT SUBMIT without a detailed description of
test_train_fairseq.
"""


import argparse
import sys
import os
import math
import collections
from datetime import datetime

pytorch_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
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
collate_tokens_gpu = data_utils.collate_tokens
import train as fairseq_train


def collate_tokens_new(values,
                       pad_idx,
                       eos_idx=None,
                       left_pad=False,
                       move_eos_to_beginning=False):
  """Copied over from fairseq.data_utils, and modified so that num_columns in the output tensor is not too variable."""
  # correcting columns
  global PAD_TO_LENGTH
  size = max(v.size(0) for v in values)
  if size > PAD_TO_LENGTH:
    print(
        'I had to change PAD_TO_LENGTH from {} to {}, this is going to trigger graph recompiles'
        .format(PAD_TO_LENGTH, size))
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

from fairseq import options, tasks, checkpoint_utils, progress_bar, utils
from fairseq.trainer import Trainer
from fairseq.data import iterators
from fairseq.meters import StopwatchMeter, AverageMeter


def parse_args():
  # We need to control certain flags here.
  # e.g. parallelization needs to be suppressed and deferred to torch_xla flags
  # e.g. input tensor shapes need to be controlled via
  #   max_sentences, required_batch_size_multiple
  parser = options.get_training_parser()
  parser.add_argument('--num_cores', type=int, default=8)
  parser.add_argument('--pad_to_length', type=int, default=64)
  parser.add_argument('--log_steps', type=int, default=20)
  parser.add_argument('--use_gpu', action='store_true')
  parser.add_argument('--metrics_debug', action='store_true')
  FLAGS = options.parse_args_and_arch(parser)
  if not FLAGS.use_gpu:
    if FLAGS.update_freq != [1]:
      FLAGS.update_freq = [1]
      print(('overriding update_freq. It is now globally 1.'
             ' Gradient update delaying is achieved through'
             ' `num_cores` in TPU setting.'))
    if FLAGS.fp16:
      print('suppressing "fp16" as this is controlled by env var XLA_USE_BF16')
      FLAGS.fp16 = False
    if FLAGS.clip_norm == 0.0:
      print('clip_norm needs to be nonzero for good TPU performance, setting it to 25')
      FLAGS.clip_norm = 25.0
    if FLAGS.distributed_world_size > 1:
      print('suppressing "distributed_world_size"')
      FLAGS.distributed_world_size = 1
    if FLAGS.distributed_init_method is not None:
      print('suppressing "distributed_init_method"')
      FLAGS.distributed_init_method = None
    if FLAGS.max_sentences != FLAGS.required_batch_size_multiple:
      batch_size = max(
          filter(lambda r: r is not None,
                 [FLAGS.max_sentences, FLAGS.required_batch_size_multiple]))
      print('"max_sentences" and "required_batch_size_multiple" must be equal'
            ' to have good performance on TPUs. Using {}'.format(batch_size))
      FLAGS.max_sentences = batch_size
      FLAGS.required_batch_size_multiple = batch_size
    if FLAGS.max_sentences_valid is not None and FLAGS.max_sentences_valid != FLAGS.max_sentences:
      FLAGS.max_sentences_valid = FLAGS.max_sentences
      print('"max_sentences_valid" and "max_sentences" must be equal'
            ' to have good performance on TPUs. Using {}'.format(
                FLAGS.max_sentences))
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
      lambda: task.build_model(args), device_ids=DEVICES)
  criteria = {
      device: task.build_criterion(args)
      for device in model_parallel._device_ids
  }
  models = {
      model_parallel._get_model_device(model): model
      for model in model_parallel._models
  }
  model, criterion = model_parallel._models[0], list(criteria.values())[0]
  print(model)
  print('| model {}, criterion {}'.format(args.arch,
                                          criterion.__class__.__name__))
  print('| num. model params: {} (num. trained: {})'.format(
      sum(p.numel() for p in model.parameters()),
      sum(p.numel() for p in model.parameters() if p.requires_grad),
  ))

  # Build trainers
  trainers = {
      device: Trainer(args, task, models[device], criteria[device])
      for device in model_parallel._device_ids
  }
  trainer = trainers[DEVICES[0]]
  lr = trainer.get_lr()

  # TODO(taylanbil): for now, this next line is only creating the iterator.
  # validate its behavior with the case where a checkpoint actually exists.

  # Load the latest checkpoint if one is available and restore the
  # corresponding train iterator
  extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
  valid_subsets = args.valid_subset.split(',')
  return task, trainers, model_parallel, epoch_itr, lr, valid_subsets


def now():
  return datetime.now().strftime('%H:%M:%S')


def main_tpu(args):

  def log_step(step_type, device, step, tracker=None, metrics_debug=False):
    msg = '{}/ {}, device {}, step {}'.format(step_type, now(), device, step)
    if tracker:
      msg += ', Rate={:.2f}'.format(tracker.rate())
    if metrics_debug:
      msg += ', Compiles={}, _local_scalar_dense={}'.format(
          count_compiles(),
          torch_xla._XLAC._xla_counter_value('aten::_local_scalar_dense'))
    return msg

  def train_loop_fn(model, loader, device, context):
    trainer = trainers[str(device)]
    stats = None
    tracker = xm.RateTracker()
    for i, samples in loader:
      if not (i % args.log_steps):
        print(log_step('training', device, i, tracker=tracker, metrics_debug=args.metrics_debug))
      _log_output = trainer.train_step(samples)
      xm.optimizer_step(trainer.optimizer)
      tracker.add(len(samples) * BATCH_SIZE)
    stats = fairseq_train.get_training_stats(trainer)
    return tracker, stats

  def valid_loop_fn(model, loader, device, context):
    trainer = trainers[str(device)]
    # reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss']:
      meter = trainer.get_meter(k)
      if meter is not None:
        meter.reset()
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for i, sample in loader:
      if not (i % args.log_steps):
        print(log_step('validation', device, i, tracker=None, metrics_debug=args.metrics_debug))
      log_output = trainer.valid_step(sample)
      for k, v in log_output.items():
        if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
          continue
        extra_meters[k].update(v)
    stats = fairseq_train.get_valid_stats(trainer)
    for k, meter in extra_meters.items():
      stats[k] = meter.avg
    return stats

  def validate_subset(args, trainers, task, epoch_itr, subset):
    print('Validating the subset "{}"'.format(subset))
    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            list(trainers.values())[0].get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_workers=args.num_workers).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args,
        itr,
        epoch_itr.epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple')
    stats_per_device = model_parallel(valid_loop_fn, progress)
    valid_losses = [stats['loss'].avg for stats in stats_per_device]
    print('validation stats on subset "{}" - {}'.format(subset, now()))
    for stats in stats_per_device:
      # print(now())
      # progress.print(stats, tag=subset, step=trainer.get_num_updates())
      # print(now())
      pass
    return valid_losses

  def validate(args, trainers, task, epoch_itr, subsets):
    valid_losses = {
        subset: validate_subset(args, trainers, task, epoch_itr, subset)
        for subset in subsets}
    return valid_losses

  def initialize_loader_for_epoch(args, epoch_itr):
    if epoch_itr.epoch <= len(args.update_freq):
      update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
      update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False, shuffle=(epoch_itr.epoch >= args.curriculum))
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple')
    return progress

  def keep_training(lr, epoch_itr, trainers):
    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = min(trainer.get_lr() for trainer in trainers.values())
    n_updates = max(trainer.get_num_updates() for trainer in trainers.values())
    return ((lr > FLAGS.min_lr) and (epoch_itr.epoch < max_epoch) and
            (n_updates < max_update))

  print('Args')
  for key, val in args.__dict__.items():
    print('\t{} {}'.format(key, val))
  print('---------')

  task, trainers, model_parallel, epoch_itr, lr, valid_subsets = prepare_task(
      args)

  train_meter = StopwatchMeter()
  train_meter.start()
  while keep_training(lr, epoch_itr, trainers):
    # TRAINING
    print('Epoch {} begin {}'.format(epoch_itr.epoch + 1, now()))
    progress = initialize_loader_for_epoch(args, epoch_itr)
    out = model_parallel(train_loop_fn, progress)
    trackers, stats_ = zip(*out)
    print('Epoch {} Training stats:'.format(epoch_itr.epoch))
    for device, trainer in trainers.items():
      stats = fairseq_train.get_training_stats(trainer)
      #print('device {}'.format(device))
      #progress.print(stats, tag=device)
    print('Epoch {} Tracker Rates:'.format(epoch_itr.epoch))
    for tracker in trackers:
      print('\tRate={:.2f}'.format(tracker.rate()))
    print('Epoch {} end {}'.format(epoch_itr.epoch, now()))
    if args.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

    # VALIDATION
    valid_losses = [None]
    if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
      valid_losses = validate(args, trainers, task, epoch_itr, valid_subsets)

      # FIXME(taylanbil): computing validation losses is VERY slow! 
      #   probably due to stats['loss'].avg implementation.
      #   do this in the xla land and return the value?

      # only use average first validation loss from the first device
      # to update the learning rate
      vloss = valid_losses[valid_subsets[0]][0]
      print('old learning rate: {}'.format(lr))
      lr = trainers[DEVICES[0]].lr_step(epoch_itr.epoch, vloss)
      print('new learning rate: {}'.format(lr))

      # save checkpoint
      if epoch_itr.epoch % args.save_interval == 0:
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, vloss)

    if args.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())

  train_meter.stop()
  print('| done training in {:.1f} seconds'.format(train_meter.sum))


def count_compiles():
  metricsreport = torch_xla._XLAC._xla_metrics_report().split('\n')
  for i, line in enumerate(metricsreport):
    if line.endswith('CompileTime'):
      return int(''.join([c for c in metricsreport[i + 1] if c.isdigit()]))


if __name__ == '__main__':
  # override certain args so that we use XLA parallelism instead of torch.
  FLAGS = parse_args()
  if FLAGS.use_gpu:
    data_utils.collate_tokens = collate_tokens_gpu
    fairseq_train.cli_main()
  else:
    BATCH_SIZE = FLAGS.max_sentences
    DEVICES = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)
    PAD_TO_LENGTH = FLAGS.pad_to_length
    main_tpu(FLAGS)
