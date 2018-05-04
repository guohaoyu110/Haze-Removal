from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

import tensorflow as tf
import numpy as np

import input_output_SYNTHIA_RAND_CITYSCAPES

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 4,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('batch_size_eval', 1,
                            """Number of images to process in a batch for evaluation.""")
tf.app.flags.DEFINE_string('hazy_data_dir', '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/Hazy_daytime_trainvaltest',
                           """Path to the SYNTHIA_RAND_CITYSCAPES hazy data directory.""")
tf.app.flags.DEFINE_string('clean_data_dir', '/scratch_net/nowin/csakarid/Code/Deep_learning/Toyota-foggy/data/SYNTHIA_RAND_CITYSCAPES/RGB_daytime_trainvaltest',
                           """Path to the SYNTHIA_RAND_CITYSCAPES original, clean data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = input_output_SYNTHIA_RAND_CITYSCAPES.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = input_output_SYNTHIA_RAND_CITYSCAPES.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 40.0	  # Staircase "width" for learning rate decay.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
MU = 0.9                          # Momentum.
WEIGHT_DECAY = 0.0001             # Weight decay factor for regularization.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# List of GPUs used for training
GPU_IDs = [int(gpu_id) for gpu_id in os.environ['SGE_GPU'].split('\n')]


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_on_gpu(name, shape, initializer):#, gpu_id):
  """Helper to create a Variable stored on GPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  #with tf.device('/gpu:%d' % gpu_id):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

  # Ensure proper weight initialization.
  # Rescale the standard deviation of the truncated normal distribution
  # for initial weight values based on the number of inputs
  # of the unit that is considered.
  number_of_unit_inputs = shape[0] * shape[1] * shape[2]
  stddev = np.sqrt(2.0 / number_of_unit_inputs)
  
  #var = _variable_on_cpu(name, shape,
  #                       tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  var = _variable_on_gpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))#,
                         #gpu_id)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(hazy_images):
  """Build the single-scale dehazing model that treats RGB color channels separately.

  Args:
    hazy_images: Images returned from input().

  Returns:
    Predictions for clean version of input hazy_images.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  # Split color channels.
  hazy_images_red, hazy_images_green, hazy_images_blue = tf.split(3, 3, hazy_images)

  # conv1
  with tf.variable_scope('conv1') as scope:

    # Red channel.
    with tf.device('/gpu:%d' % GPU_IDs[0]):
      kernel_red = _variable_with_weight_decay('weights_red',
                                               shape=[5, 5, 1, 8],
                                               wd=WEIGHT_DECAY)
      conv_red = tf.nn.conv2d(hazy_images_red, kernel_red, [1, 1, 1, 1], padding='SAME')
      biases_red = _variable_on_gpu('biases_red', [8], tf.constant_initializer(0.0))
      pre_activation_red = tf.nn.bias_add(conv_red, biases_red)
      conv1_red = tf.nn.relu(pre_activation_red, name='conv1_red')
      _activation_summary(conv1_red)

    # Green channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(1, len(GPU_IDs) - 1)]):
      kernel_green = _variable_with_weight_decay('weights_green',
                                                 shape=[5, 5, 1, 8],
                                                 wd=WEIGHT_DECAY)
      conv_green = tf.nn.conv2d(hazy_images_green, kernel_green, [1, 1, 1, 1], padding='SAME')
      biases_green = _variable_on_gpu('biases_green', [8], tf.constant_initializer(0.0))
      pre_activation_green = tf.nn.bias_add(conv_green, biases_green)
      conv1_green = tf.nn.relu(pre_activation_green, name='conv1_green')
      _activation_summary(conv1_green)

    # Blue channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(2, len(GPU_IDs) - 1)]):
      kernel_blue = _variable_with_weight_decay('weights_blue',
                                                shape=[5, 5, 1, 8],
                                                wd=WEIGHT_DECAY)
      conv_blue = tf.nn.conv2d(hazy_images_blue, kernel_blue, [1, 1, 1, 1], padding='SAME')
      biases_blue = _variable_on_gpu('biases_blue', [8], tf.constant_initializer(0.0))
      pre_activation_blue = tf.nn.bias_add(conv_blue, biases_blue)
      conv1_blue = tf.nn.relu(pre_activation_blue, name='conv1_blue')
      _activation_summary(conv1_blue)

  # conv2
  with tf.variable_scope('conv2') as scope:

    # Red channel.
    with tf.device('/gpu:%d' % GPU_IDs[0]):
      kernel_red = _variable_with_weight_decay('weights_red',
                                               shape=[5, 5, 8, 16],
                                               wd=WEIGHT_DECAY)
      conv_red = tf.nn.conv2d(conv1_red, kernel_red, [1, 1, 1, 1], padding='SAME')
      biases_red = _variable_on_gpu('biases_red', [16], tf.constant_initializer(0.0))
      pre_activation_red = tf.nn.bias_add(conv_red, biases_red)
      conv2_red = tf.nn.relu(pre_activation_red, name='conv2_red')
      _activation_summary(conv2_red)

    # Green channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(1, len(GPU_IDs) - 1)]):
      kernel_green = _variable_with_weight_decay('weights_green',
                                                 shape=[5, 5, 8, 16],
                                                 wd=WEIGHT_DECAY)
      conv_green = tf.nn.conv2d(conv1_green, kernel_green, [1, 1, 1, 1], padding='SAME')
      biases_green = _variable_on_gpu('biases_green', [16], tf.constant_initializer(0.0))
      pre_activation_green = tf.nn.bias_add(conv_green, biases_green)
      conv2_green = tf.nn.relu(pre_activation_green, name='conv2_green')
      _activation_summary(conv2_green)

    # Blue channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(2, len(GPU_IDs) - 1)]):
      kernel_blue = _variable_with_weight_decay('weights_blue',
                                                shape=[5, 5, 8, 16],
                                                wd=WEIGHT_DECAY)
      conv_blue = tf.nn.conv2d(conv1_blue, kernel_blue, [1, 1, 1, 1], padding='SAME')
      biases_blue = _variable_on_gpu('biases_blue', [16], tf.constant_initializer(0.0))
      pre_activation_blue = tf.nn.bias_add(conv_blue, biases_blue)
      conv2_blue = tf.nn.relu(pre_activation_blue, name='conv2_blue')
      _activation_summary(conv2_blue)

  # conv3
  with tf.variable_scope('conv3') as scope:

    # Red channel.
    with tf.device('/gpu:%d' % GPU_IDs[0]):
      kernel_red = _variable_with_weight_decay('weights_red',
                                               shape=[5, 5, 16, 32],
                                               wd=WEIGHT_DECAY)
      conv_red = tf.nn.conv2d(conv2_red, kernel_red, [1, 1, 1, 1], padding='SAME')
      biases_red = _variable_on_gpu('biases_red', [32], tf.constant_initializer(0.0))
      pre_activation_red = tf.nn.bias_add(conv_red, biases_red)
      conv3_red = tf.nn.relu(pre_activation_red, name='conv3_red')
      _activation_summary(conv3_red)

    # Green channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(1, len(GPU_IDs) - 1)]):
      kernel_green = _variable_with_weight_decay('weights_green',
                                                 shape=[5, 5, 16, 32],
                                                 wd=WEIGHT_DECAY)
      conv_green = tf.nn.conv2d(conv2_green, kernel_green, [1, 1, 1, 1], padding='SAME')
      biases_green = _variable_on_gpu('biases_green', [32], tf.constant_initializer(0.0))
      pre_activation_green = tf.nn.bias_add(conv_green, biases_green)
      conv3_green = tf.nn.relu(pre_activation_green, name='conv3_green')
      _activation_summary(conv3_green)

    # Blue channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(2, len(GPU_IDs) - 1)]):
      kernel_blue = _variable_with_weight_decay('weights_blue',
                                                shape=[5, 5, 16, 32],
                                                wd=WEIGHT_DECAY)
      conv_blue = tf.nn.conv2d(conv2_blue, kernel_blue, [1, 1, 1, 1], padding='SAME')
      biases_blue = _variable_on_gpu('biases_blue', [32], tf.constant_initializer(0.0))
      pre_activation_blue = tf.nn.bias_add(conv_blue, biases_blue)
      conv3_blue = tf.nn.relu(pre_activation_blue, name='conv3_blue')
      _activation_summary(conv3_blue)

  # linear
  with tf.variable_scope('linear') as scope:

    # Red channel.
    with tf.device('/gpu:%d' % GPU_IDs[0]):
      kernel_red = _variable_with_weight_decay('weights_red',
                                               shape=[1, 1, 32, 1],
                                               wd=WEIGHT_DECAY)
      conv_red = tf.nn.conv2d(conv3_red, kernel_red, [1, 1, 1, 1], padding='SAME')
      biases_red = _variable_on_gpu('biases_red', [1], tf.constant_initializer(0.0))
      linear_red = tf.nn.bias_add(conv_red, biases_red, name='linear_red')

    # Green channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(1, len(GPU_IDs) - 1)]):
      kernel_green = _variable_with_weight_decay('weights_green',
                                                 shape=[1, 1, 32, 1],
                                                 wd=WEIGHT_DECAY)
      conv_green = tf.nn.conv2d(conv3_green, kernel_green, [1, 1, 1, 1], padding='SAME')
      biases_green = _variable_on_gpu('biases_green', [1], tf.constant_initializer(0.0))
      pre_activation_green = tf.nn.bias_add(conv_green, biases_green)
      linear_green = tf.nn.bias_add(conv_green, biases_green, name='linear_green')

    # Blue channel.
    with tf.device('/gpu:%d' % GPU_IDs[min(2, len(GPU_IDs) - 1)]):
      kernel_blue = _variable_with_weight_decay('weights_blue',
                                                shape=[1, 1, 32, 1],
                                                wd=WEIGHT_DECAY)
      conv_blue = tf.nn.conv2d(conv3_blue, kernel_blue, [1, 1, 1, 1], padding='SAME')
      biases_blue = _variable_on_gpu('biases_blue', [1], tf.constant_initializer(0.0))
      pre_activation_blue = tf.nn.bias_add(conv_blue, biases_blue)
      linear_blue = tf.nn.bias_add(conv_blue, biases_blue, name='linear_blue')

    # Concatenate predictions.
    linear = tf.concat(3, [linear_red, linear_green, linear_blue], name=scope.name)
    _activation_summary(linear)

  return linear


def loss(clean_images_predicted, clean_images_ground_truth):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    clean_images_predicted: Predicted images from inference(). 4-D tensor
      of shape [batch_size, image_height, image_width, 3]
    clean_images_ground_truth: Ground truth clean images. 4-D tensor
      of shape [batch_size, image_height, image_width, 3]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average data loss across the batch.
  data_loss = tf.reduce_mean(tf.square(tf.subtract(clean_images_predicted, clean_images_ground_truth)),
                             name = 'data_loss')
  tf.add_to_collection('losses', data_loss)

  # The total loss is defined as the data loss plus all of the weight
  # decay terms (regularization loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def data_loss(clean_images_predicted, clean_images_ground_truth):
  """Compute L2 loss for the predicted images.

  Args:
    clean_images_predicted: Predicted images from inference(). 4-D tensor
      of shape [batch_size, image_height, image_width, 3]
    clean_images_ground_truth: Ground truth clean images. 4-D tensor
      of shape [batch_size, image_height, image_width, 3]

  Returns:
    Loss tensor of type float.
  """
  data_loss = tf.reduce_mean(tf.square(tf.subtract(clean_images_predicted, clean_images_ground_truth)))

  return data_loss


def _add_loss_summaries(total_loss):
  """Add summaries for losses in single-scale baseline model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train single-scale baseline model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients using update with momentum.
  with tf.control_dependencies([loss_averages_op]):
    # opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.MomentumOptimizer(lr, MU)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def input(val):
  """Construct input for SYNTHIA_RAND_CITYSCAPES training and evaluation.

  Args:
    val: bool, indicating if one should use the train or val data set.

  Returns:
    hazy_images: Images. 4D tensor of [batch_size, height, width, 3] size.
    clean_images: Images. 4D tensor of [batch_size, height, width, 3] size.
    hazy_image_filenames: Filenames. 1D tensor of [batch_size] size.
  """

  hazy_data_dir = FLAGS.hazy_data_dir
  clean_data_dir = FLAGS.clean_data_dir
  batch_size = FLAGS.batch_size if not val else FLAGS.batch_size_eval
  hazy_images, clean_images, hazy_image_filenames = input_output_SYNTHIA_RAND_CITYSCAPES.input_pipeline(val,
                                                    hazy_data_dir, clean_data_dir, batch_size)

  if FLAGS.use_fp16:
    hazy_images = tf.cast(hazy_images, tf.float16)
    clean_images = tf.cast(clean_images, tf.float16)

  return hazy_images, clean_images, hazy_image_filenames


def encode_prediction_png(clean_images_predicted):

  clean_images_uint8 = input_output_SYNTHIA_RAND_CITYSCAPES.postprocess_uint8_format(clean_images_predicted)
  images_png_encoded = tf.image.encode_png(clean_images_uint8[0])

  return images_png_encoded

