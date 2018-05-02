from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import single_scale_RGB_independent_colors as model_spec

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/results/SYNTHIA_RAND_CITYSCAPES/single_scale_RGB_independent_colors/layers_3-one_fil_5_dep_8-two_fil_5_dep_16-three_fil_5_dep_32/configuration_01/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 61225,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('val', False,
                         """Whether to use validation set.""")


def train():
  """Train SYNTHIA_RAND_CITYSCAPES for a number of steps."""
  with tf.Graph().as_default():
    
    # Fix the graph-level random seed for reproducibility across
    # different runs.
    tf.set_random_seed(20)
    
    global_step = tf.Variable(0, trainable=False)

    # Get hazy and clean images for SYNTHIA.
    val = FLAGS.val
    hazy_images, clean_images_ground_truth, _ = model_spec.input(val)

    # Build a Graph that computes the dehazed predictions from the
    # inference model.
    clean_images_predicted = model_spec.inference(hazy_images)

    # Calculate loss.
    loss = model_spec.loss(clean_images_predicted, clean_images_ground_truth)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model_spec.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.5f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 200 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
