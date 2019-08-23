# -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.python.ops import image_ops

def read_and_decode_train(record):
    keys_to_features={
        'Image/Label' : tf.FixedLenFeature((), tf.int64),
        'Image/Raw' : tf.FixedLenFeature((), tf.string),
        'Image/Height' : tf.FixedLenFeature((1,), tf.int64),
        'Image/Width' : tf.FixedLenFeature((1,), tf.int64)}
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(parsed['Image/Raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [32, 32, 3])
    label = tf.cast(parsed['Image/Label'], tf.int32)
    height = tf.cast(parsed['Image/Height'], tf.int32)
    width = tf.cast(parsed['Image/Width'], tf.int32)

    label = tf.one_hot(label, 10, 1, 0)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.resize_images(image, [40,40])
    image = tf.image.random_crop(image,[32,32,3])

    return image, label, height, width

def read_and_decode_val(record):
    keys_to_features={
        'Image/Label' : tf.FixedLenFeature((), tf.int64),
        'Image/Raw' : tf.FixedLenFeature((), tf.string),
        'Image/Height' : tf.FixedLenFeature((1,), tf.int64),
        'Image/Width' : tf.FixedLenFeature((1,), tf.int64)}
    parsed = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(parsed['Image/Raw'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [32, 32, 3])
    label = tf.cast(parsed['Image/Label'], tf.int32)
    height = tf.cast(parsed['Image/Height'], tf.int32)
    width = tf.cast(parsed['Image/Width'], tf.int32)
    label = tf.one_hot(label, 10, 1, 0)

    return image, label, height, width

def inputs(data_path, batch_size, is_training = True):
    with tf.name_scope('tfrecord_input') as scope:
        dataset = tf.data.TFRecordDataset(data_path, buffer_size=batch_size*3)
        if is_training:
            dataset = dataset.map(read_and_decode_train)
        else:
            dataset = dataset.map(read_and_decode_val)
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        image, label, height, width = iterator.get_next()

    return iterator, image, label, height, width