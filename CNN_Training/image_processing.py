# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read and preprocess image data.

 Image processing occurs on a single image at a time. Image are read and
 preprocessed in parallel across multiple threads. The resulting images
 are concatenated together to form a single batch for training or evaluation.

 -- Provide processed image data for a network:
 inputs: Construct batches of evaluation examples of images.
 distorted_inputs: Construct batches of training examples of images.
 batch_inputs: Construct batches of training or evaluation examples of images.

 -- Data processing:
 parse_example_proto: Parses an Example proto containing a training example
   of an image.

 -- Image decoding:
 decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

 -- Image preprocessing:
 image_preprocessing: Decode and preprocess one image for evaluation or training
 distort_image: Distort one image for training a network.
 eval_image: Prepare one image for evaluation.
 distort_color: Distort the color in one image for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.signal as sp
import random
'''
###########################
variable
###########################
'''
_addVar = 0  # if we add variations as blur, noise etc effects
'''
###########################
function
###########################
'''


def inputs(dataset, batch_size, height, width, channels, num_preprocess_threads=4):
    """Generate batches of ImageNet images for evaluation.

  Use this function as the inputs for evaluating a network.

  Note that some (minimal) image preprocessing occurs during evaluation
  including central cropping and resizing of the image to fit the network.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       image_size, 3].
    labels: 1-D integer Tensor of [FLAGS.batch_size].
  """
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset,
            batch_size,
            height,
            width,
            channels,
            add_variations=False,
            train=False,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=4)

    return images, labels


def distorted_inputs(dataset, batch_size, height, width, channels, add_variations, num_preprocess_threads=4):
    """Generate batches of distorted versions of ImageNet images.

  Use this function as the inputs for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    dataset: instance of Dataset class specifying the dataset.
    batch_size: integer, number of examples in batch
    num_preprocess_threads: integer, total number of preprocessing threads but
      None defaults to FLAGS.num_preprocess_threads.

  Returns:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
  """

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset,
            batch_size,
            height,
            width,
            channels,
            add_variations,
            train=True,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=4)
    return images, labels


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
  Returns:
    color-distorted image
  """
    with tf.name_scope(scope, 'distort_color', [image]):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=62. / 255.)
            #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            #image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=62. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            #image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            #image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_motion_blur(image, height, width, lenth, thread_id=0, scope=None):
    """
  added code by angnong:
  Distort the image with blutting effect
  """

    # generate filter kernel
    # set horizontal blurring
    kernel = np.ones(lenth)
    # kernel_3d = np.repeat(kernel.reshape(1,lenth,1), 3, axis=2)

    # print(kernel_3d.shape)
    filter_m = kernel.reshape(1, lenth, 1, 1)

    # expand image dims to 4D
    image = tf.expand_dims(image, 0)
    # applying the kernel to every channel of the image
    image_r, image_g, image_b = tf.split(image, 3, 3)

    # do conv2d
    image_r = tf.nn.conv2d(image_r, filter_m, [1, 1, 1, 1], padding='SAME')
    image_g = tf.nn.conv2d(image_g, filter_m, [1, 1, 1, 1], padding='SAME')
    image_b = tf.nn.conv2d(image_b, filter_m, [1, 1, 1, 1], padding='SAME')

    # tf.nn.conv2d(image)
    # image = tf.nn.conv2d(image,filter_m,[1,1,1,1],padding='SAME')
    # resize to 3 dimension
    image_r = tf.squeeze(image_r, [0, 3])
    image_g = tf.squeeze(image_g, [0, 3])
    image_b = tf.squeeze(image_b, [0, 3])

    image = tf.stack([image_r, image_g, image_b], axis=2)

    return image


def distort_scale_blur(image, height, width):

    image = tf.image.resize_images(image, [int(0.4 * height), int(0.4 * width)])
    image = tf.image.resize_images(image, [height, width])

    return image


def distort_noise(image, height, width, dens):
    """
  code added by angnong:
  Distort one image with Gaussian noise
  """
    mean = 0
    var = 0.01
    sigma = var**0.5
    # gauss_init = np.zeros((height, width, 3))
    # gauss = np.random.normal(mean, sigma, (height, width, 3))
    # gauss = gauss.reshape(height, width, 3)
    # # keep density procent number of the noise image
    # dens_num = int(dens * height * width)
    #
    #
    #
    # for i in range(0, dens_num):
    #     rand_x = random.randint(0, height - 1)
    #     rand_y = random.randint(0, width - 1)
    #     gauss_init[rand_x, rand_y, :] = gauss[rand_x, rand_y, :]

    noise = tf.random_normal(shape=[height, width, 3], stddev=sigma)
    select = tf.to_float(tf.less(tf.random_uniform(shape=[height, width, 1], minval=0, maxval=1), dens))
    select = tf.concat([select, select, select], axis=2)

    image = tf.add(image, noise * select)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_image(image, bbox, height, width, add_variations, thread_id=0, scope=None):
    """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
    """ Change added by angnong:
    1. specify the color/saturation/contrast range
    2. add noise
    3. add blurring
    4. random choose one from three effects(color, noise, blurring) or no effect
  """
    with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # Display the bounding box in the first thread only.
        if not thread_id:
            tf.summary.image('original_image', tf.expand_dims(image, 0))

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # Note added by angnong:
        # Images in CASIA-WebFace are already in a fine face region, so we don't need
        # use face detection to cut a bbox
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an allowed
        # range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.

        image = tf.image.crop_to_bounding_box(image,
                                              tf.to_int32(bbox[0, 0, 0]),
                                              tf.to_int32(bbox[0, 0, 1]),
                                              tf.to_int32(bbox[0, 0, 2]), tf.to_int32(bbox[0, 0, 3]))
        #image = image[bbox[0,0,0]: bbox[0,0,2],bbox[0,0,1]: bbox[0,0,3] ,:]

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant(0, shape=[1, 0, 4]),
            min_object_covered=0.85,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image('images_with_distorted_bounding_box', image_with_distorted_box)

        # Crop the image to the specified bounding box.
        distorted_image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(distorted_image, [height, width], method=resize_method)
        # distorted_image = tf.image.resize_images(
        #    image, [height, width], method=resize_method)

        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.summary.image('cropped_resized_image', tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        if not thread_id:
            tf.summary.image('horizontally_flipped', tf.expand_dims(distorted_image, 0))

        # TODO: ADD more variations or not (50% chance)
        if add_variations:
            with tf.name_scope('distort_color'):
                # Randomly distort the colors.
                prob_color = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image = tf.cond(
                    tf.equal(prob_color, 0), lambda: distort_color(distorted_image, thread_id), lambda: distorted_image)

            with tf.name_scope('distort_blur'):
                #randomly distort the blurring
                prob_blur_type = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image_blur = tf.cond(
                    tf.equal(prob_blur_type, 0),
                    lambda: distort_motion_blur(distorted_image, height, width, 5, thread_id),
                    lambda: distort_scale_blur(distorted_image, height, width))

                prob_blur = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image_blur = tf.cond(
                    tf.equal(prob_blur, 0), lambda: distorted_image_blur, lambda: distorted_image)

            with tf.name_scope('distort_noise'):
                # distort image with gaussian noise
                prob_noise = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image = tf.cond(
                    tf.equal(prob_noise, 0), lambda: distort_noise(distorted_image, height, width, 0.2),
                    lambda: distorted_image)

            # add a small summary
            if not thread_id:
                tf.summary.image('final_image', tf.expand_dims(distorted_image, 0))
    return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.

  Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def image_preprocessing(image_buffer, bbox, height, width, channels, add_variations, train, thread_id=0):
    """Decode and preprocess one image for evaluation or training.
  code edited by angnong: add filename to find the bbox
  Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """

    image = decode_jpeg(image_buffer)
    if train:
        image = distort_image(image, bbox, height, width, add_variations, thread_id)
    else:
        image = eval_image(image, height, width)

    if channels == 1:
        image = tf.image.rgb_to_grayscale(image)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def parse_example_proto(example_serialized):

    # return tf.train.Example(features=tf.train.Features(feature={
    #     'image/width':
    #     tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    #     'image/height':
    #     tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
    #     'image/label/values':
    #     tf.train.Feature(int64_list=tf.train.Int64List(value=label_values)),
    #     'image/label/names':
    #     tf.train.Feature(bytes_list=tf.train.BytesList(value=label_names)),
    #     'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(
    #         value=[filename.encode('utf-8')])),
    #     'image/encoded':
    #     tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
    # }))

    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/label/values': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/label/names': tf.VarLenFeature(dtype=tf.string),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto
    feature_map.update(
        {k: sparse_float32
         for k in ['image/bbox/xmin', 'image/bbox/ymin', 'image/bbox/width', 'image/bbox/height']})

    features = tf.parse_single_example(example_serialized, feature_map)

    image = decode_jpeg(features['image/encoded'])

    im_height = tf.shape(image)[0]
    im_width = tf.shape(image)[1]

    xmin = tf.expand_dims(features['image/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/bbox/ymin'].values, 0)
    width = tf.expand_dims(features['image/bbox/width'].values, 0)
    height = tf.expand_dims(features['image/bbox/height'].values, 0)
    #xmax = xmin + width
    #ymax = ymin + height

    # change value to [0,1)
    #xmin = tf.div(tf.to_float(xmin), tf.to_float(im_width))
    #ymin = tf.div(tf.to_float(ymin), tf.to_float(im_height))
    #xmax = tf.div(tf.to_float(xmax), tf.to_float(im_width))
    #ymax = tf.div(tf.to_float(ymax), tf.to_float(im_height))

    # Note that we impose an ordering of (y,x) just to make life difficult
    # WTF?
    bbox = tf.concat([ymin, xmin, height, width], 0)

    # Force the variable number of bounding boxes into the shape
    #[1,num_boxes,coords]
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    return features['image/encoded'], tf.cast(
        features['image/label/values'], dtype=tf.int32), features['image/label/names'], bbox


def batch_inputs(dataset,
                 batch_size,
                 height,
                 width,
                 channels,
                 add_variations,
                 train,
                 num_preprocess_threads=4,
                 num_readers=4):
    """Contruct batches of training or evaluation examples from the image dataset.

  Args:
    dataset: instance of Dataset class specifying the dataset.
      See dataset.py for details.
    batch_size: integer
    train: boolean
    num_preprocess_threads: integer, total number of preprocessing threads
    num_readers: integer, number of parallel readers

  Returns:
    images: 4-D float Tensor of a batch of images
    labels: 1-D integer Tensor of [batch_size].

  Raises:
    ValueError: if data is not found
  """
    with tf.name_scope('batch_processing'):
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
            num_readers = 8

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * 16
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples, dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size, dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = dataset.reader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, _, bbox = parse_example_proto(example_serialized)

            image = image_preprocessing(image_buffer, bbox, height, width, channels, add_variations, train, thread_id)
            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, channels])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, tf.reshape(label_index_batch, [batch_size])
