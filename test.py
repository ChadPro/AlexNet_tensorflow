import tensorflow as tf
from nets import alexnet_224

img = tf.ones([1,224,224,3])
net = alexnet_224.alexnet_net(img)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    r_net = sess.run(net)
    print r_net.shape
