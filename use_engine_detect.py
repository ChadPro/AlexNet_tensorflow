# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorrt as trt
import tensorflow as tf
import numpy as np
import time
from os.path import join
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

SOURCE_PATH = "demo/17flowers/"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

start_time = time.time()
with open("alexnet.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(engine[0])), dtype=trt.nptype(engine.get_binding_dtype(engine[0])))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(engine[1])), dtype=trt.nptype(engine.get_binding_dtype(engine[1])))

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

with engine.create_execution_context() as context:
    dur_time = time.time() - start_time
    print("### Load Use %f mniutes" %(dur_time/60.))
    start_time = time.time()
    for j in range(100):
        for i in range(1):
            image_path = SOURCE_PATH + str(i) + ".jpg"
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224,224), cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2,0,1))
            img = np.reshape(img, [3*224*224])
            h_input = np.ascontiguousarray(img, dtype=np.float32)
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async(batch_size=1, bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            if j == 99:
                print("########## Detect use tensorrt uff ############")
                output = np.frombuffer(h_output, dtype=np.float32, count=len(h_output))
                # print(np.argmax(output))
                print(output)
    dur_time = time.time() - start_time
    print("### Detect Use %f mniutes" %(dur_time/60.))