# -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.
import tensorflow as tf
import numpy as np
import time
import cv2
import tensorrt as trt
from os.path import join
import pycuda.autoinit
import pycuda.driver as cuda
from nets import nets_factory

UFF_PATH = "model.uff"
MODEL_SAVE_PATH = "./models/model.ckpt"

def use_uff_create():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder:
        with builder.create_network() as network:
            with trt.UffParser() as parser:
                parser.register_input("inputdata", (3, 224, 224))
                parser.register_output("outputdata")
                parser.parse(UFF_PATH, network)

                builder.max_batch_size = 1
                builder.max_workspace_size = 1 << 20
                with builder.build_cuda_engine(network) as engine:
                    serialized_engine = engine.serialize()
                    with trt.Runtime(TRT_LOGGER) as runtime:
                        engine = runtime.deserialize_cuda_engine(serialized_engine)
                        with open("alexnet.engine", "wb") as f:
                            f.write(engine.serialize())
                            print("ok")

                    input_shape = engine.get_binding_shape(engine[0])
                    output_shape = engine.get_binding_shape(engine[1])
                    print(input_shape)
                    print(output_shape)

def use_ckpt_create():
    x = tf.placeholder(tf.float32, [1, 224, 224, 3], name='x-input')
    alexnet = nets_factory.get_network("alexnet_224")
    y = alexnet.alexnet_net(x, num_classes=17, regularizer=None, is_dropout=False)

    # print("")
    # variables = tf.global_variables()
    # for var in variables:
    #     print(var)
    # print("")

    conv_1_w = tf.get_default_graph().get_tensor_by_name("conv_1/weights:0") # [11,11,3,96]
    conv_1_b = tf.get_default_graph().get_tensor_by_name("conv_1/bias:0") # [96]
    conv_2_w = tf.get_default_graph().get_tensor_by_name("conv_2/weights:0")
    conv_2_b = tf.get_default_graph().get_tensor_by_name("conv_2/bias:0")
    conv_3_w = tf.get_default_graph().get_tensor_by_name("conv_3/weights:0")
    conv_3_b = tf.get_default_graph().get_tensor_by_name("conv_3/bias:0")
    conv_4_w = tf.get_default_graph().get_tensor_by_name("conv_4/weights:0")
    conv_4_b = tf.get_default_graph().get_tensor_by_name("conv_4/bias:0")
    conv_5_w = tf.get_default_graph().get_tensor_by_name("conv_5/weights:0")
    conv_5_b = tf.get_default_graph().get_tensor_by_name("conv_5/bias:0")
    fc_1_w = tf.get_default_graph().get_tensor_by_name("fc1/weights:0")
    fc_1_b = tf.get_default_graph().get_tensor_by_name("fc1/bias:0")
    fc_2_w = tf.get_default_graph().get_tensor_by_name("fc2/weights:0")
    fc_2_b = tf.get_default_graph().get_tensor_by_name("fc2/bias:0")
    output_w = tf.get_default_graph().get_tensor_by_name("output/weights:0")
    output_b = tf.get_default_graph().get_tensor_by_name("output/bias:0")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_SAVE_PATH)
        r_conv_1_w, r_conv_1_b = sess.run([conv_1_w, conv_1_b])
        r_conv_2_w, r_conv_2_b = sess.run([conv_2_w, conv_2_b])
        r_conv_3_w, r_conv_3_b = sess.run([conv_3_w, conv_3_b])
        r_conv_4_w, r_conv_4_b = sess.run([conv_4_w, conv_4_b])
        r_conv_5_w, r_conv_5_b = sess.run([conv_5_w, conv_5_b])
        r_fc_1_w, r_fc_1_b = sess.run([fc_1_w, fc_1_b])
        r_fc_2_w, r_fc_2_b = sess.run([fc_2_w, fc_2_b])
        r_output_w, r_output_b = sess.run([output_w, output_b])


    def pad_size(W, F, S, padding="SAME"):
        pad_top = (0,0)
        pad_down = (0,0)
        if padding == "SAME":
            new_size = int(W / S) + 1 if W % S > 0 else int(W / S)
            pad_needed = (new_size - 1) * S + F - W
            pad_top = int(pad_needed / 2)
            pad_down = pad_needed - pad_top
        return (pad_top, pad_top), (pad_down, pad_down)
            
    def trt_conv2d(network, inputs, dw_size, strides, conv_w_value, conv_b_value, activation=True, padding="SAME"):
        W = inputs.shape[-1]
        F = dw_size[0]
        output_maps = dw_size[-1]
        kernel_shape = (F, F)
        net_stride = (strides[1], strides[2])

        conv_w_value = np.expand_dims(conv_w_value, axis=0)
        conv_w_value = np.transpose(conv_w_value, (0,4,3,1,2))
        conv_w_value = np.ascontiguousarray(conv_w_value, dtype=np.float32)
        conv_b_value = np.ascontiguousarray(conv_b_value, dtype=np.float32)

        pre_pad, post_pad = pad_size(W, F, net_stride[0])
        net_pad= network.add_padding(input=inputs, pre_padding=pre_pad, post_padding=post_pad)
        net = network.add_convolution(input=net_pad.get_output(0), num_output_maps=output_maps, kernel_shape=kernel_shape, kernel=conv_w_value, bias=conv_b_value)
        net.stride = net_stride
        net = network.add_activation(net.get_output(0), trt.ActivationType.RELU)

        return net.get_output(0)

    def trt_fc(network, inputs, num_outputs, fc_w_value, fc_b_value, activation=True, flatten=False, padding="SAME"):
        fc_w_value = np.transpose(fc_w_value, (1,0))
        fc_w_value = np.ascontiguousarray(fc_w_value, dtype=np.float32)
        fc_b_value = np.ascontiguousarray(fc_b_value, dtype=np.float32)

        net = network.add_fully_connected(inputs, num_outputs=num_outputs, kernel=fc_w_value)
        # if activation:
        net = network.add_activation(net.get_output(0), trt.ActivationType.RELU)
        return net.get_output(0)      

    def trt_maxpool(network, inputs, pool_size, strides, padding="SAME"):
        W = inputs.shape[-1]
        F = pool_size[1]
        win_size = (F, F)
        net_stride = (strides[1], strides[2])

        pre_pad, post_pad = pad_size(W, F, net_stride[0])
        net_pad= network.add_padding(input=inputs, pre_padding=pre_pad, post_padding=post_pad)
        net = network.add_pooling(input=net_pad.get_output(0), type=trt.PoolingType.MAX, window_size=win_size)
        net.stride = net_stride

        return net.get_output(0)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder:
        with builder.create_network() as network:
            print("TensorRT version : ", trt.__version__)

            input_tensor = network.add_input(name="inputdata", dtype=trt.float32, shape=[3,224,224])

            net = trt_conv2d(network, input_tensor, r_conv_1_w.shape, [1,4,4,1], r_conv_1_w, r_conv_1_b)
            net = trt_maxpool(network, net, [1,3,3,1], [1,2,2,1])
            net = trt_conv2d(network, net, r_conv_2_w.shape, [1,1,1,1], r_conv_2_w, r_conv_2_b)
            net = trt_maxpool(network, net, [1,3,3,1], [1,2,2,1])
            net = trt_conv2d(network, net, r_conv_3_w.shape, [1,1,1,1], r_conv_3_w, r_conv_3_b)
            net = trt_conv2d(network, net, r_conv_4_w.shape, [1,1,1,1], r_conv_4_w, r_conv_4_b)
            net = trt_conv2d(network, net, r_conv_5_w.shape, [1,1,1,1], r_conv_5_w, r_conv_5_b)
            net = trt_maxpool(network, net, [1,3,3,1], [1,2,2,1])   # (256,7,7) value right

            print(r_fc_1_w[0, 100:130])
            net = trt_fc(network, net, 4096, r_fc_1_w, r_fc_1_b)    # (12544, 4096)
            # net = trt_fc(network, net, r_fc_2_w.shape[-1], r_fc_2_w, r_fc_2_b)
            # net = trt_fc(network, net, r_output_w.shape[-1], r_output_w, r_output_b, activation=False)

            print("output shape : ", net.shape)
            net.name = "outputdata"
            network.mark_output(tensor=net)

            builder.max_batch_size = 1
            builder.max_workspace_size = 1 << 20

            with builder.build_cuda_engine(network) as engine:
                serialized_engine = engine.serialize()
                with trt.Runtime(TRT_LOGGER) as runtime:
                    engine = runtime.deserialize_cuda_engine(serialized_engine)
                    with open("alexnet_test.engine", "wb") as f:
                        f.write(engine.serialize())
                        print("ok")

def main():
    use_uff_create()
    # use_ckpt_create()

if __name__ == "__main__":
    main()