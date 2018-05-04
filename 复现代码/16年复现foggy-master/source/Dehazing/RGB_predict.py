from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import glob

import numpy as np
import tensorflow as tf

import single_scale_RGB_independent_colors as model_spec

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('results_dir', '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/results/SYNTHIA_RAND_CITYSCAPES/single_scale_RGB_independent_colors/layers_3-one_fil_5_dep_8-two_fil_5_dep_16-three_fil_5_dep_32/configuration_01/results/test/beta_0.07',
                           """Directory where to write prediction results.""")
tf.app.flags.DEFINE_boolean('val', True,
                         """Whether to use validation set.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/results/SYNTHIA_RAND_CITYSCAPES/single_scale_RGB_independent_colors/layers_3-one_fil_5_dep_8-two_fil_5_dep_16-three_fil_5_dep_32/configuration_01/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 1195,
                            """Number of examples to run.""")


def predict_and_write_results(saver, images_png_encoded, hazy_image_filenames):
  """Predict clean images for entire split and write results.

  Args:
    saver: Saver.
    images_png_encoded: Op for string with PNG-encoded dehazed image.
    hazy_image_filenames: Op for string with filename of hazy counterpart
      of the output PNG-encoded dehazed image.
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

      num_iter = FLAGS.num_examples
      step = 0
      examples_count = 0
      print('Starting prediction...')
      while step < num_iter and not coord.should_stop():
        png_string, filename = sess.run([images_png_encoded, hazy_image_filenames])
        examples_count += np.size(png_string)
        hazy_filename = filename[0]
        dehazed_filename_tmp = hazy_filename.replace('hazy', 'dehazed')
        dehazed_filename = dehazed_filename_tmp.replace(FLAGS.hazy_data_dir + '/test/beta_0.07', FLAGS.results_dir)
        with open(dehazed_filename, 'wb') as f:
          f.write(png_string)
        step += 1

      print('Finished writing prediction results for %d images.' % examples_count)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def predict():
  """Predict clean SYNTHIA images using final model."""
  with tf.Graph().as_default() as g:
    
    # Get hazy and clean images for SYNTHIA validation set.
    val = FLAGS.val
    hazy_images, _, hazy_image_filenames = model_spec.input(val)

    # Build a Graph that computes the dehazed predictions from the
    # inference model.
    clean_images_predicted = model_spec.inference(hazy_images)

    # Postprocess the predictions and encode them in PNG format.
    images_png_encoded = model_spec.encode_prediction_png(clean_images_predicted)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model_spec.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    predict_and_write_results(saver, images_png_encoded, hazy_image_filenames)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.results_dir):
    tf.gfile.DeleteRecursively(FLAGS.results_dir)
  tf.gfile.MakeDirs(FLAGS.results_dir)
  predict()


if __name__ == '__main__':
  tf.app.run()
