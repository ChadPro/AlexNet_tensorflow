import tensorflow as tf
import numpy as np
from nets import alexnet_224_test
import cv2

IMG_PATH = "demo/17flowers/1.jpg"
MODEL_SAVE_PATH = "model/model.ckpt"

x = tf.placeholder(tf.float32, [1, 224, 224, 3], name='x-input')
y = alexnet_224_test.alexnet_net(x, num_classes=17, is_training=False, regularizer=None, is_dropout=False)
# output_y = tf.nn.softmax(y, name="outputdata")
fc_1_w = tf.get_default_graph().get_tensor_by_name("fc1/weights:0")
fc_1_b = tf.get_default_graph().get_tensor_by_name("fc1/bias:0")

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, MODEL_SAVE_PATH)

    img = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (224,224), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)

    r = sess.run(y, feed_dict={x : img})
    r = r[0]
    print(r.shape)
    print(r[0:20])
    # print(r)

    # fcwv, fcbv = sess.run([fc_1_w, fc_1_b])
    # print(fcwv.shape)
    # print(fcwv[0, 100:130])
    # print(fcbv.shape)
    # print(fcbv[100:130])