# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
'''
本脚本 alexnet_net() 实现了论文中 alexnet net
'''
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from collections import namedtuple

#input & output param
IMAGE_SIZE = 224
NUM_CHANNELS = 3
STDDEV = 0.01
VGG_MEAN = [122.173, 116.150, 103.504]  # bgr
DEFAULT_OUTPUT_NODE = 1000
BN_DECAY = 0.9
ACTIVATION = tf.nn.relu

def conv2d_block(inputs, dw_size, strides, downsample=False, is_training=True, padding="SAME", scope=""):
    _stride = strides
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        conv_deep = dw_size[-1]
        conv_weights = tf.get_variable("weights", dw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        conv_biases = tf.get_variable("bias", conv_deep, initializer=tf.constant_initializer(0.))
        conv2d = tf.nn.conv2d(inputs, conv_weights, strides=_stride, padding=padding)
        net = ACTIVATION(tf.nn.bias_add(conv2d, conv_biases))
    return net

def maxpool_block(inputs, pool_size, strides, downsample=True, is_training=True, padding="SAME", scope=""):
    _stride = strides
    with tf.name_scope(scope):
        pool = tf.nn.max_pool(inputs, ksize=pool_size, strides=_stride, padding=padding)
    return pool

def fc_block(inputs, outputs, regularizer, activation=None, flatten=False, is_dropout=False, is_training=True, scope=""):
    if flatten:
        net_shape = inputs.get_shape()
        nodes = tf.math.multiply(tf.math.multiply(net_shape[1], net_shape[2]), net_shape[3])
        reshaped = tf.reshape(inputs, [net_shape[0], nodes])
        inputs = reshaped
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        fc_weights = tf.get_variable("weights", [inputs.get_shape()[1], outputs], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc_weights))
        fc_biases = tf.get_variable("bias", [outputs], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        fc = tf.nn.bias_add(tf.matmul(inputs, fc_weights), fc_biases)
        if activation:
            fc = activation(fc)
        if is_dropout:
            fc = tf.cond(is_training, lambda: tf.nn.dropout(fc, 0.5), lambda: fc)
            # fc = tf.nn.dropout(fc, 0.5)
    return fc

def alexnet_net(inputs, \
                num_classes=DEFAULT_OUTPUT_NODE, \
                is_training=True, \
                reuse=None, \
                white_bal=False, \
                regularizer=None, \
                is_dropout=False, \
                scope='alexnet_net_224_original'):

    
    dst_img = inputs
    if white_bal:
        bgr_scaled = dst_img
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr_scaled)
        assert red.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
        assert green.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
        assert blue.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, 1]
        dst_img = tf.concat(axis=3, values=[blue - VGG_MEAN[0],green - VGG_MEAN[1],red - VGG_MEAN[2],])
        assert dst_img.get_shape().as_list()[1:] == [IMAGE_SIZE, IMAGE_SIZE, 3]

    with tf.name_scope(scope):
        net = conv2d_block(dst_img, [11,11,3,96], [1,4,4,1], is_training=is_training, scope="conv_1")
        net = maxpool_block(net, [1,3,3,1], [1,2,2,1], is_training=is_training, scope="pool_1")
        net = conv2d_block(net, [5,5,96,256], [1,1,1,1], is_training=is_training, scope="conv_2")
        net = maxpool_block(net, [1,3,3,1], [1,2,2,1], is_training=is_training, scope="pool_2")
        net = conv2d_block(net, [3,3,256,384], [1,1,1,1], is_training=is_training, scope="conv_3")
        net = conv2d_block(net, [3,3,384,384], [1,1,1,1], is_training=is_training, scope="conv_4")
        net = conv2d_block(net, [3,3,384,256], [1,1,1,1], is_training=is_training, scope="conv_5")
        net = maxpool_block(net, [1,3,3,1], [1,2,2,1], is_training=is_training, scope="pool_3")

        net = fc_block(net, 4096, regularizer, activation=ACTIVATION, flatten=True, is_dropout=is_dropout, is_training=is_training, scope="fc1")
        net = fc_block(net, 4096, regularizer, activation=ACTIVATION, is_dropout=is_dropout, is_training=is_training, scope="fc2")
        net = fc_block(net, num_classes, regularizer, is_training=is_training, scope="output")

    return net

