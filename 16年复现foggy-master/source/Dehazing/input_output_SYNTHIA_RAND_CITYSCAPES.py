from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Global constants describing the data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2449
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 419
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 1195
IMAGE_HEIGHT = 760
IMAGE_WIDTH = 1280

def preprocess_zerocenter_and_unit_range(image_raw):
  
  # Cast the image to float32 format for subsequent processing.
  image_float = tf.cast(image_raw, tf.float32)
  
  # Linear map of original [0, 255] range of RGB values to [-0.5, 0.5].
  image_normalized = tf.subtract(tf.truediv(image_float, tf.constant(255.0)),
                                 tf.constant(0.5))
  
  return image_normalized


def postprocess_uint8_format(image_normalized):

  # Clip the image to [-0.5, 0.5] interval to avoid overflow or underflow
  # in subsequent casting.
  image_clipped = tf.maximum(tf.minimum(image_normalized, tf.constant(0.5)),
                             tf.constant(-0.5))
  
  # Linear map of transformed [-0.5, 0.5] range of RGB values to original [0, 255].
  image_original_range = tf.multiply(tf.add(image_clipped, tf.constant(0.5)),
                                     tf.constant(255.0))

  # Round each RGB value to the nearest integer before casting, as the latter
  # truncates the decimal part of float32 values.
  image_rounded = tf.round(image_original_range)
  
  # Cast the image to uint8 format for subsequent PNG encoding.
  image_uint8 = tf.cast(image_rounded, tf.uint8)
  
  return image_uint8


def read_synthia(filename_pairs_queue):
  """Reads and parses examples from SYNTHIA images in PNG format.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_pairs_queue: A queue of string pairs with the filenames to read from.

  Returns:
    hazy_image: 3D tensor of shape [height, width, 3] and type uint8.
    clean_image: 3D tensor of shape [height, width, 3] and type uint8.
    hazy_image_filename: 0D tensor of string type.
  """
  #######  WholeFileReader approach: NOT working
  # Read whole .png image files, getting filenames from the filename_pairs_queue.
  #reader = tf.WholeFileReader()
  #hazy_image_filename, hazy_image_file_contents = reader.read(filename_pairs_queue[0])
  #hazy_image = tf.image.decode_png(hazy_image_file_contents)
  #_, clean_image_file_contents = reader.read(filename_pairs_queue[1])
  #clean_image = tf.image.decode_png(clean_image_file_contents)
  #######

  ####### read_file approach
  hazy_image_file_contents = tf.read_file(filename_pairs_queue[0])
  hazy_image = tf.image.decode_png(hazy_image_file_contents)
  hazy_image_filename = filename_pairs_queue[0]

  clean_image_file_contents = tf.read_file(filename_pairs_queue[1])
  clean_image = tf.image.decode_png(clean_image_file_contents)
  
  return hazy_image, clean_image, hazy_image_filename


def generate_image_pairs_batch(hazy_image, clean_image, hazy_image_filename, min_queue_examples,
                                   batch_size, shuffle):
  """Construct a queued batch of image pairs and hazy image filenames (hazy image,
     clean image, hazy filename).

  Args:
    hazy_image: 3D tensor of [height, width, 3] size and type float32.
    clean_image: 3D tensor of [height, width, 3] size and type float32.
    hazy_image_filename: 0D tensor of string type.
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    hazy_images: Images. 4D tensor of [batch_size, height, width, 3] size.
    clean_images: Images. 4D tensor of [batch_size, height, width, 3] size.
    hazy_image_filenames: Filenames. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' hazy + clean images from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    hazy_images, clean_images, hazy_image_filenames = tf.train.shuffle_batch(
        [hazy_image, clean_image, hazy_image_filename],
        batch_size=batch_size,
        shapes=[[IMAGE_HEIGHT, IMAGE_WIDTH, 3], [IMAGE_HEIGHT, IMAGE_WIDTH, 3], []],
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    hazy_images, clean_images, hazy_image_filenames = tf.train.batch(
        [hazy_image, clean_image, hazy_image_filename],
        batch_size=batch_size,
        shapes=[[IMAGE_HEIGHT, IMAGE_WIDTH, 3], [IMAGE_HEIGHT, IMAGE_WIDTH, 3], []],
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training image pairs in the visualizer.
  tf.image_summary('hazy_images', hazy_images)
  tf.image_summary('clean_images', clean_images)

  return hazy_images, clean_images, hazy_image_filenames


def input_pipeline(val, hazy_data_dir, clean_data_dir, batch_size):
  """Construct input for SYNTHIA as (hazy image, clean image, hazy filename) triplets.

  Args:
    val: bool, indicating whether to use the validation or the training set.
    hazy_data_dir: Path to the hazy data directory containing the subdirectories
      for the hazy part of training and validation sets.
    clean_data_dir: Path to the clean data directory containing the subdirectories
      for the clean part of training and validation sets.
    batch_size: Number of images per batch.

  Returns:
    hazy_images: Images. 4D tensor of [batch_size, height, width, 3] size.
    clean_images: Images. 4D tensor of [batch_size, height, width, 3] size.
    hazy_image_filenames: Filenames. 1D tensor of [batch_size] size.
  """
  if not val:
    # TensorFlow function to generate lists with image filenames. The generated lists
    # are not guaranteed to have the image filenames in exactly the same (sorted) order,
    # which results in ERRONEOUS pairs of hazy input and clean ground truth
    #hazy_filenames = tf.train.match_filenames_once(hazy_data_dir + '/train/*.png')
    #clean_filenames = tf.train.match_filenames_once(clean_data_dir + '/train/*.png')
    
    # Custom Python code to generate "aligned" lists with image filenames.
    hazy_filenames = sorted(glob.glob(hazy_data_dir + '/train/*.png'))
    clean_filenames = sorted(glob.glob(clean_data_dir + '/train/*.png'))
    assert len(hazy_filenames) == len(clean_filenames)
    assert len(hazy_filenames) == NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    shuffle = True
  else:
    # Custom Python code to generate "aligned" lists with image filenames.
    hazy_filenames = sorted(glob.glob(hazy_data_dir + '/test/beta_0.07/*.png'))
    clean_filenames = sorted(glob.glob(clean_data_dir + '/test/*.png'))
    assert len(hazy_filenames) == len(clean_filenames)
    #assert len(hazy_filenames) == NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    #num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    assert len(hazy_filenames) == NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TEST
    shuffle = False

  # Create a queue that produces the filenames to read.
  filename_pairs_queue = tf.train.slice_input_producer([hazy_filenames, clean_filenames])

  # Read examples from files in the filename queue.
  hazy_image_raw, clean_image_raw, hazy_image_filename = read_synthia(filename_pairs_queue)

  # Preprocess the images.
  hazy_image_normalized = preprocess_zerocenter_and_unit_range(hazy_image_raw)
  clean_image_normalized = preprocess_zerocenter_and_unit_range(clean_image_raw)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.2
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d SYNTHIA image pairs. '
         'This may take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return generate_image_pairs_batch(hazy_image_normalized, clean_image_normalized, hazy_image_filename,
                                    min_queue_examples, batch_size, shuffle)

