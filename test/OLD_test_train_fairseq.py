# DEFAULT_KWARGS = dict(
#     activation_dropout=0.0,
#     activation_fn='relu',
#     adam_betas='(0.9, 0.98)',
#     adam_eps=1e-08,
#     adaptive_input=False,
#     adaptive_softmax_cutoff=None,
#     adaptive_softmax_dropout=0,
#     arch='transformer_vaswani_wmt_en_de_big',
#     attention_dropout=0.1,
#     bucket_cap_mb=25,
#     clip_norm=0.0,
#     cpu=False,
#     criterion='label_smoothed_cross_entropy',
#     curriculum=0,
#     data='/mnt/data/dummy_fairseq',
#     dataset_impl='cached',
#     ddp_backend='c10d',
#     decoder_attention_heads=16,
#     decoder_embed_dim=1024,
#     decoder_embed_path=None,
#     decoder_ffn_embed_dim=4096,
#     decoder_input_dim=1024,
#     decoder_layers=6,
#     decoder_learned_pos=False,
#     decoder_normalize_before=False,
#     decoder_output_dim=1024,
#     device_id=0,
#     distributed_backend='nccl',
#     distributed_init_method='tcp://10.138.0.17:8085',
#     distributed_no_spawn=False,
#     distributed_port=8085,
#     distributed_rank=0,
#     distributed_world_size=1,
#     dropout=0.3,
#     encoder_attention_heads=16,
#     encoder_embed_dim=1024,
#     encoder_embed_path=None,
#     encoder_ffn_embed_dim=4096,
#     encoder_layers=6,
#     encoder_learned_pos=False,
#     encoder_normalize_before=False,
#     fix_batches_to_gpus=False,
#     fp16=True,
#     fp16_init_scale=128,
#     fp16_scale_tolerance=0.0,
#     fp16_scale_window=None,
#     keep_interval_updates=-1,
#     keep_last_epochs=-1,
#     label_smoothing=0.1,
#     lazy_load=False,
#     left_pad_source='True',
#     left_pad_target='False',
#     log_format=None,
#     log_interval=100,
#     lr=[0.0005],
#     lr_scheduler='inverse_sqrt',
#     max_epoch=0,
#     max_sentences=None,
#     max_sentences_valid=None,
#     max_source_positions=1024,
#     max_target_positions=1024,
#     max_tokens=3584,
#     max_update=100000,
#     memory_efficient_fp16=False,
#     min_loss_scale=0.0001,
#     min_lr=1e-09,
#     no_epoch_checkpoints=False,
#     no_progress_bar=True,
#     no_save=True,
#     no_token_positional_embeddings=False,
#     num_workers=0,
#     optimizer='adam',
#     optimizer_overrides='{}',
#     raw_text=False,
#     required_batch_size_multiple=8,
#     reset_lr_scheduler=False,
#     reset_optimizer=False,
#     restore_file='checkpoint_last.pt',
#     save_dir='checkpoints',
#     save_interval=1,
#     save_interval_updates=16000,
#     seed=3,
#     sentence_avg=False,
#     share_all_embeddings=True,
#     share_decoder_input_output_embed=False,
#     skip_invalid_size_inputs_valid_test=False,
#     source_lang='en',
#     target_lang='de',
#     task='translation',
#     tensorboard_logdir='',
#     threshold_loss_scale=None,
#     update_freq=[1],
#     upsample_primary=16,
#     validate_interval=1,
#     warmup_init_lr=1e-07,
#     warmup_updates=4000,
#     weight_decay=0.0)

# Setup import folders.
import sys
import argparse
pytorch_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
xla_folder = os.path.join(pytorch_folder, 'xla')
sys.path.insert(0, xla_folder)
sys.path.insert(0, pytorch_folder)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--logdir', type=str, default=None)
parser.add_argument('--num_cores', type=int, default=8)
#parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--metrics_debug', action='store_true')

import torch
from common_utils import TestCase, run_tests
from fairseq import tasks  # , utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq import optim
import torch_xla
import torch_xla_py.data_parallel as dp
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm


def build_optimizer(args, model):
  params = list(filter(lambda p: p.requires_grad, model.parameters()))
  return optim.build_optimizer(args, params)


def train_loop_fn(model, loader, device):
  criterion = task.build_criterion(args)
  tracker = xm.RateTracker()
  tracker.add(FLAGS.batch_size)
  optimizer = build_optimizer(args, model)
  for i, samples in loader:
    print('Processing minibatch:%d for device %s' % (i, device.index))
    task.train_step(samples[0], model, criterion, optimizer, False)
    xm.optimizer_step(optimizer)
    # print("Rate: {}".format(tracker.rate()))
    if x % FLAGS.log_steps == 0:
      args = device, x, loss.item(), tracker.rate()
      print('[{}]({}) Training Loss={:.5f} Rate={:.2f}'.format(*args))


def test_loop_fn(model, loader, device, context):
  pass
  # for i, samples in loader:
  #   output = model(data)
  #   pred = output.max(1, keepdim=True)[1]
  #   correct += pred.eq(target.view_as(pred)).sum().item()
  #   total_samples += data.size()[0]
  #   if x % FLAGS.log_steps == 0:
  #     args = device, x, loss.item(), tracker.rate())
  #     print('[{}]({}) Training Loss={:.5f} Rate={:.2f}'.format(*args)
  # return


def parse_args(args):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'


def initialize_fairseq():
  task = tasks.setup_task(FLAGS)
  task.load_dataset('train', combine=True, epoch=0)
  task.load_dataset('test', combine=True, epoch=0)
  model = lambda: task.build_model(FLAGS)
  return task, model


def train_fairseq():
  task, model = initialize_fairseq()
  devices = xm.get_xla_supported_devices(max_devices=FLAGS.num_cores)
  model_parallel = dp.DataParallel(model, device_ids=devices)
  max_positions = (1024, 1024)
  epoch_itr = task.get_batch_iterator(
      dataset=task.dataset('train'),
      max_tokens=FLAGS.max_tokens,
      max_sentences=FLAGS.max_sentences,
      max_positions=(1024, 1024)  # TODO: improve
      ignore_invalid_inputs=True,
      required_batch_size_multiple=FLAGS.required_batch_size_multiple,
      seed=FLAGS.seed,
      num_shards=FLAGS.distributed_world_size,
      shard_id=FLAGS.distributed_rank,
      num_workers=FLAGS.num_workers,
  )
  raise

  update_freq = (
    args.update_freq[epoch_itr.epoch - 1]
    if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1])

  # Initialize data iterator
  itr = epoch_itr.next_epoch_itr(
      fix_batches_to_gpus=False,
      shuffle=(epoch_itr.epoch >= args.curriculum),
  )
  # Create the iterator for training
  train_loader = iterators.GroupedIterator(itr, update_freq)
  for epoch in range(1, FLAGS.num_epochs + 1):
    print("Begin epoch {}".format(epoch))
    model_parallel(train_loop_fn, train_loader)
    # accuracies = model_parallel(test_loop_fn, test_loader)
    # accuracy = sum(accuracies) / len(devices)
    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())
    print("End epoch {}".format(epoch))


# ######################################
class TrainFAIRSEQModel(TestCase):

  def tearDown(self):
    super(TrainFAIRSEQModel, self).tearDown()

  def test_accurracy(self):
    # self.assertGreaterEqual(train_fairseq(), FLAGS.target_accuracy)
    pass


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
# TODO: run tests
#run_tests()
