# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import tensorflow as tf
import time
from nets import nets_factory
from datasets import dataset_factory
from py_extend import vgg_acc

#训练参数
REGULARIZATION_RATE= 0.0001    
MOVING_AVERAGE_DECAY = 0.99   

####################
#   Learn param    #
####################
tf.app.flags.DEFINE_string('train_data_path','', 'Dataset for train.')
tf.app.flags.DEFINE_string('val_data_path', '', 'Dataset for val.')
tf.app.flags.DEFINE_integer('num_classes', 17, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Default batch_size 64.')
tf.app.flags.DEFINE_integer('data_nums', 0, 'All train data nums.')
tf.app.flags.DEFINE_integer('epoch', 100, 'Train epoches.')
tf.app.flags.DEFINE_bool('mul_gpu', False, 'Right chose use more gpu.')
tf.app.flags.DEFINE_string('gpu_id', '', 'TF Device list.')

tf.app.flags.DEFINE_float('learning_rate_base', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.99, 'Decay learning rate.')
tf.app.flags.DEFINE_integer('learning_decay_step', 500, 'Learning rate decay step.')
tf.app.flags.DEFINE_float('gpu_fraction', 0.9, 'How to use gpu.')
tf.app.flags.DEFINE_string('train_model_dir', './model/model.ckpt', 'Directory where checkpoints are written to.')
tf.app.flags.DEFINE_string('log_dir', './log_dir', 'Log file saved.')

tf.app.flags.DEFINE_string('dataset', 'flowers17_dataset', 'Chose dataset in dataset_factory.')
tf.app.flags.DEFINE_bool('white_bal',False, 'If white balance.')
tf.app.flags.DEFINE_bool('regularizer', False, 'If use regularizer.')
tf.app.flags.DEFINE_bool('dropout', False, 'If use dropout.')
tf.app.flags.DEFINE_integer('image_size', 224, 'Default image size.')

tf.app.flags.DEFINE_string('net_chose','alexnet_224', 'Use to chose net.')
tf.app.flags.DEFINE_bool('fine_tune', False, 'Is fine_tune work.')
tf.app.flags.DEFINE_string('restore_model_dir', '', 'Restore model.')
FLAGS = tf.app.flags.FLAGS

# 每一个epoch需要的step等于数据总量除一个step上的batch_size
step_per_epoch = int(FLAGS.data_nums / FLAGS.batch_size)
total_steps = step_per_epoch * FLAGS.epoch

# 记录最高分数
f = open("model/scores.txt", "w")
f.write("30.0\n")
f.close()

if len(FLAGS.gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

########################################
#           Train function             #
########################################
def train():
    #1. 获取网络 及 输入
    with tf.name_scope("Data_Input"):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3], name='x-input')
        y_ = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.num_classes], name='y-input')
        x_val = tf.placeholder(tf.float32, [1, FLAGS.image_size, FLAGS.image_size, 3], name='x-input-val')
        y_val = tf.placeholder(tf.float32, [1, FLAGS.num_classes], name='y-input-val')
        
    #2. 前向传播 + 计算目标函数
    with tf.name_scope("Forward_Propagation_calc_loss"):
        alexnet = nets_factory.get_network(FLAGS.net_chose)
        if FLAGS.regularizer:
            print("###################### 优化方法 Regularizer#########################")
            print("===>> 使用优化方法 Regularizer")
            regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE, scope="regularizer")  
        else:
            regularizer = None

        loss_list = []
        if not FLAGS.mul_gpu:
            y = alexnet.alexnet_net(x, num_classes=FLAGS.num_classes, regularizer=regularizer, is_dropout=FLAGS.dropout)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
            loss_list.append(cross_entropy)
        else:
            pass
            gpu_num = len(FLAGS.gpu_id.strip().split(','))
            gpu_id = range(gpu_num)
            x_split = tf.split(x, gpu_num)
            y_split_ = tf.split(y_, gpu_num)
        
            for i, d in enumerate(gpu_id):
                with tf.device("/gpu:%s"%d):
                    with tf.name_scope("%s_%s"%("tower", d)):
                        y = alexnet.alexnet_net(x_split[i], num_classes=FLAGS.num_classes, regularizer=regularizer, is_dropout=FLAGS.dropout)
                        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_split_[i], logits=y)
                        loss_list.append(cross_entropy)
        
        val_y = alexnet.alexnet_net(x_val, num_classes=FLAGS.num_classes, is_training=False, regularizer=regularizer)
        cross_entropy_mean = tf.reduce_mean(loss_list) 
        if FLAGS.regularizer:
            loss = cross_entropy_mean + tf.reduce_mean(tf.add_n(tf.get_collection('regularizer_losses')))
        else:
            loss = cross_entropy_mean

    #3. 反向梯度
    with tf.name_scope("Back_Train"):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base ,global_step, FLAGS.learning_decay_step, FLAGS.learning_rate_decay)  
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 

    #4. 记录
    with tf.name_scope("Summary_Saver"):
        tf.summary.image("input", x, 5)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())
        saver = tf.train.Saver()  

    #5. 获取数据
    with tf.name_scope("Get_Data"):
        alexnet_dataset = dataset_factory.get_dataset(FLAGS.dataset)
        train_iterator, input_X, input_Y, _, __ = alexnet_dataset.inputs(FLAGS.train_data_path, FLAGS.batch_size, is_training=True)
        val_iterator, input_X_val, input_Y_val, _, __ = alexnet_dataset.inputs(FLAGS.val_data_path, 1, is_training=False)

    #6. 开启会话    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    with tf.Session(config=config) as sess:
        # init global variables  
        sess.run(tf.global_variables_initializer()) 
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)
        if (len(FLAGS.restore_model_dir) > 0) & FLAGS.fine_tune:
            print("#####=============> Restore Model : "+str(FLAGS.restore_model_dir))
            saver.restore(sess, FLAGS.restore_model_dir)
        else:
            print("#####=============> Create Model : "+str(FLAGS.train_model_dir))

        startTime = time.time()
        print("##### Total epoches = %d" %FLAGS.epoch)
        print("##### Total steps = %d" %total_steps)
        for i in range(total_steps):
            X_input, Y_input = sess.run([input_X, input_Y])
            summary_str, _, loss_value, step = sess.run([merged, train_step, loss, global_step], feed_dict={x:X_input, y_:Y_input})   
            writer.add_summary(summary_str, i)

            if i % step_per_epoch == 0:
                learn_rate_now = FLAGS.learning_rate_base * ( FLAGS.learning_rate_decay**(step/ FLAGS.learning_decay_step))

                l_outy_val = []
                l_outy__val = []
                for j in range(100):
                    X_input_val, Y_input_val = sess.run([input_X_val, input_Y_val])
                    outy_val, outy__val = sess.run([val_y, y_val], feed_dict={x_val:X_input_val, y_val:Y_input_val})
                    l_outy_val.append(outy_val)
                    l_outy__val.append(outy__val)
                l_outy_val = np.array(l_outy_val)
                l_outy__val = np.array(l_outy__val)
                l_outy_val = np.squeeze(l_outy_val, axis=1)
                l_outy__val = np.squeeze(l_outy__val, axis=1)

                acc_val_top1 = vgg_acc.acc_top1(l_outy_val, l_outy__val)
                acc_val_top5 = vgg_acc.acc_top5(l_outy_val, l_outy__val)
                run_time = time.time() - startTime
                run_time = run_time / 60

                print("############ epoch : %d ################"%int(i / step_per_epoch))
                print("   learning_rate = %g                    "%learn_rate_now)
                print("   lose(batch)   = %g                    "%loss_value)
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
        minuteTime = durationTime / 60
        print("To train the AlexNet, we use %d minutes" %minuteTime)


def main(argv=None):
    train()

if __name__== '__main__': 
    tf.app.run()
