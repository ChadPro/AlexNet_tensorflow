 # -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.
import tensorflow as tf
from nets import alexnet_32

img = tf.ones([1,32,32,3])
net = alexnet_32.alexnet_net(img, num_classes=10, is_training=False)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    r_net = sess.run(net)
    print(r_net.shape)
