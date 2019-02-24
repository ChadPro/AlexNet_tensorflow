# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import os
import pickle
import numpy as np
import math
from PIL import Image
import tensorflow as tf
import sys
import time
from nets import nets_factory
from datasets import dataset_factory
from py_extend import vgg_acc
from keras.preprocessing.image import ImageDataGenerator

#训练参数
REGULARIZATION_RATE= 0.0001    
MOVING_AVERAGE_DECAY = 0.99   

####################
#   Learn param    #
####################
tf.app.flags.DEFINE_float('learning_rate_base', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.99, 'Decay learning rate.')
tf.app.flags.DEFINE_integer('learning_decay_step', 500, 'Learning rate decay step.')
tf.app.flags.DEFINE_integer('data_nums', 0, 'All train data nums.')
tf.app.flags.DEFINE_integer('epoch', 200, 'Train epoches.')
tf.app.flags.DEFINE_float('gpu_fraction', 0.7, 'How to use gpu.')
tf.app.flags.DEFINE_string('train_model_dir', './model/model.ckpt', 'Directory where checkpoints are written to.')
tf.app.flags.DEFINE_string('log_dir', './log_dir', 'Log file saved.')

tf.app.flags.DEFINE_string('train_data_path','', 'Dataset for train.')
tf.app.flags.DEFINE_string('val_data_path', '', 'Dataset for val.')

tf.app.flags.DEFINE_string('dataset', 'flowers17_224', 'Chose dataset in dataset_factory.')
tf.app.flags.DEFINE_bool('white_bal',False, 'If white balance.')
tf.app.flags.DEFINE_bool('regularizer', False, 'If use regularizer.')
tf.app.flags.DEFINE_bool('dropout', True, 'If use dropout.')
tf.app.flags.DEFINE_integer('image_size', 224, 'Default image size.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Default batch_size 64.')
tf.app.flags.DEFINE_integer('num_classes', 17, 'Number of classes to use in the dataset.')

tf.app.flags.DEFINE_string('net_chose','alexnet_224', 'Use to chose net.')
tf.app.flags.DEFINE_bool('fine_tune', False, 'Is fine_tune work.')
tf.app.flags.DEFINE_string('restore_model_dir', '', 'Restore model.')
FLAGS = tf.app.flags.FLAGS
step_per_epoch = int(FLAGS.data_nums / FLAGS.batch_size)
total_steps = step_per_epoch * FLAGS.epoch

########################################
#           Train function             #
########################################
def train():
    #1. 获取网络 及 输入
    with tf.name_scope("Data_Input"):
        alexnet_224 = nets_factory.get_network(FLAGS.net_chose)
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3], name='x-input')
        x_val = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3], name='x-input-val')
        rgb_img_input = tf.reverse(x, axis=[-1])  
        tf.summary.image("input", rgb_img_input, 5)
        y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_classes], name='y-input')
        y_val = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_classes], name='y-input-val')
        isTrainNow = tf.placeholder(tf.bool, name='isTrainNow')
    
    #2. 前向传播
    with tf.name_scope("Forward_Propagation"):
        if FLAGS.regularizer:
            print("#######################   Use Regularizer   #######################")
            regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE, scope="regularizer")  
        else:
            regularizer = None
        y = alexnet_224.alexnet_net(x, num_classes=FLAGS.num_classes, is_training=isTrainNow, regularizer=regularizer, is_dropout=FLAGS.dropout)
        val_y = alexnet_224.alexnet_net(x_val, num_classes=FLAGS.num_classes, is_training=isTrainNow, reuse=True, regularizer=regularizer, is_dropout=False)
        global_step = tf.Variable(0, trainable=False)

    #3. 计算损失函数
    with tf.name_scope("Calc_Loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)   
        if FLAGS.regularizer:
            loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        else:
            loss = cross_entropy_mean

    #4. 反向梯度
    with tf.name_scope("Back_Train"):
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base ,global_step, FLAGS.learning_decay_step, FLAGS.learning_rate_decay)  
        # train_step 梯度下降(学习率，损失函数，全局步数) + BN Layer Params update op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 

    #5. 计算ACC
    with tf.name_scope("Calc_Acc"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        correct_prediction_val = tf.equal(tf.argmax(val_y, 1), tf.argmax(y_val, 1))
        accuracy_val = tf.reduce_mean(tf.cast(correct_prediction_val, tf.float32))

    #6. 记录
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('acc', accuracy)
    tf.summary.scalar('val_acc', accuracy_val)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())
    saver = tf.train.Saver()  

    #7. 获取数据
    alexnet_dataset = dataset_factory.get_dataset(FLAGS.dataset)
    input_X, input_Y, testtest = alexnet_dataset.inputs(FLAGS.train_data_path,FLAGS.val_data_path,'Train', FLAGS.batch_size, None)
    input_X_val, input_Y_val, _ = alexnet_dataset.inputs(FLAGS.train_data_path,FLAGS.val_data_path,'Val', FLAGS.batch_size, None)
    datagen = ImageDataGenerator(
                            featurewise_center=False,
                            samplewise_center=False,
                            rotation_range=180,
							width_shift_range=0.1,
							height_shift_range=0.1,
							zoom_range=0.1)

    #8. 开启会话    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    with tf.Session(config=config) as sess:
        # init global variables  
        init_op = tf.global_variables_initializer()
        sess.run(init_op) 
  
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        if len(FLAGS.restore_model_dir) > 0:
            print("#####=============> Restore Model : "+str(FLAGS.restore_model_dir))
            saver.restore(sess, FLAGS.restore_model_dir)
        else:
            print("#####=============> Create Model : "+str(FLAGS.train_model_dir))

        startTime = time.time()
        print("##### epoch = %d" %FLAGS.epoch)
        print("##### total_steps = %d" %total_steps)
        for i in range(total_steps):
            X_input, Y_input, testy = sess.run([input_X, input_Y, testtest])
            gen_input = datagen.flow(X_input, batch_size=FLAGS.batch_size)
            input_imgs = next(gen_input)
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x:input_imgs, y_:Y_input, isTrainNow:True})   

            if i%step_per_epoch == 0:     
                learn_rate_now = FLAGS.learning_rate_base * ( FLAGS.learning_rate_decay**(step/ FLAGS.learning_decay_step))
                X_input_val, Y_input_val = sess.run([input_X_val, input_Y_val])
                
                summary_str, outy, outy_, outy_val, outy__val = sess.run([merged, y, y_, val_y, y_val], feed_dict={x_val:X_input_val, y_val:Y_input_val, x:X_input, y_:Y_input, isTrainNow:False})
                writer.add_summary(summary_str, i)
                acc_top1 = vgg_acc.acc_top1(outy, outy_)
                acc_top5 = vgg_acc.acc_top5(outy, outy_)
                acc_val_top1 = vgg_acc.acc_top1(outy_val, outy__val)
                acc_val_top5 = vgg_acc.acc_top5(outy_val, outy__val)
                run_time = time.time() - startTime
                run_time = run_time / 60

                print("############ epoch : %d ################"%int(i/step_per_epoch))
                print("   learning_rate = %g                    "%learn_rate_now)
                print("   lose(batch)   = %g                    "%loss_value)
                print("   acc_top1      = " + acc_top1 + "%")
                print("   acc_top5      = " + acc_top5 + "%")
                print("   acc_val_top1      = " + acc_val_top1 + "%")
                print("   acc_val_top5      = " + acc_val_top5 + "%")
                print("   train run     = %d min"%run_time)
         
                f = open("model/scores.txt", "r")
                score = float(f.readline())
                f.close()
                if float(acc_val_top1) > score :
                    print("   Net improved from score {0}%, now is {1}%".format(str(score), acc_val_top1))
                    saver.save(sess, FLAGS.train_model_dir)
                    f = open("model/scores.txt", "w")
                    f.write(acc_val_top1)
                    f.close()
                else:
                    print("   Net did not improve from score {0}%.".format(str(score)))
                print("")
                print("")

        writer.close()
        durationTime = time.time() - startTime
        minuteTime = durationTime/60
        print("To train the MobileNet, we use %d minutes" %minuteTime)
        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    train()

if __name__== '__main__': 
    tf.app.run()
