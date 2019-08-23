# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import cv2
import numpy as np
import time

SOURCE_PATH = "demo/17flowers/"

start_time = time.time()
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = './model.pb'

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        input_x = sess.graph.get_tensor_by_name("inputdata:0")
        output = sess.graph.get_tensor_by_name("outputdata:0")

        dur_time = time.time() - start_time
        print("### Load Use %f mniutes" %(dur_time/60.))

        start_time = time.time()
        for j in range(100):
            for i in range(17):
                image_path = SOURCE_PATH + str(i) + ".jpg"
                img = cv2.imread(image_path)
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.expand_dims(img, axis=0)
                yy = sess.run(output, {input_x:img})
                if j == 99:
                    print("######### Detect Result ##########")
                    print("class = {0}".format(str(np.argmax(yy))))
        dur_time = time.time() - start_time
        print("### Detect Use %f mniutes" %(dur_time/60.))