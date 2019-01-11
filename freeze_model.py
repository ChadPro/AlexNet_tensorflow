# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from nets import alexnet_224

def export_graph(model_path):
    
    #graph = tf.Graph()
    
    #with graph.as_default():
        
    input_image = tf.placeholder(tf.float32, shape=[1,224,224,3], name="inputdata")
    logits = alexnet_224.alexnet_net(input_image, num_classes=17, is_training=False)
    output_y = tf.nn.softmax(logits, name="outputdata")
        
    restore_saver = tf.train.Saver()

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        restore_saver.restore(sess, model_path)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, output_y.graph.as_graph_def(), ["outputdata"])


        with tf.gfile.GFile("./model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())

export_graph("./model/model.ckpt")



