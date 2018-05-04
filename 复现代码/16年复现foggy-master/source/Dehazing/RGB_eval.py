from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']

import numpy as np
import tensorflow as tf

import single_scale_RGB_independent_colors as model_spec

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/results/SYNTHIA_RAND_CITYSCAPES/single_scale_RGB_independent_colors/layers_3-one_fil_5_dep_8-two_fil_5_dep_16-three_fil_5_dep_32/configuration_01/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_boolean('val', True,
                         """Whether to use validation set.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/results/SYNTHIA_RAND_CITYSCAPES/single_scale_RGB_independent_colors/layers_3-one_fil_5_dep_8-two_fil_5_dep_16-three_fil_5_dep_32/configuration_01/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 419,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, loss, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    loss: Loss op.
    summary_op: Summary op.
  """

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size_eval))
      loss_accumulator = 0.0
      examples_count = 0
      step = 0
      print('Starting evaluation...')
      while step < num_iter and not coord.should_stop():
        loss_current_batch = sess.run([loss])
        loss_accumulator += np.sum(loss_current_batch)
        examples_count += np.size(loss_current_batch)
        step += 1

      # Compute loss across the entire validation set.
      validation_loss = loss_accumulator / examples_count
      print('%s: Validation Loss = %.5f' % (datetime.now(), validation_loss))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Validation Loss', simple_value=validation_loss)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval SYNTHIA for a number of steps."""
  with tf.Graph().as_default() as g:
    
    # Get hazy and clean images for SYNTHIA.
    val = FLAGS.val
    hazy_images, clean_images_ground_truth, _ = model_spec.input(val)

    # Build a Graph that computes the dehazed predictions from the
    # inference model.
    clean_images_predicted = model_spec.inference(hazy_images)

    # Calculate loss (only the data term).
    loss = model_spec.data_loss(clean_images_predicted, clean_images_ground_truth)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model_spec.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, loss, summary_op)
      if FLAGS.run_once:
        print('Finished one-off evaluation.')
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
