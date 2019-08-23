# -- coding: utf-8 --
# Copyright 2019 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time

SOURCE_PATH = "demo/17flowers/"
UFF_PATH = "model.uff"

def detect():
    print(trt.__version__)
    start_time = time.time()
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
                    print("############ engine bindings ############")
                    print(engine.get_binding_shape(engine[0]))
                    print(engine.get_binding_shape(engine[1]))

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
                            for i in range(17):
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
                                    output = np.frombuffer(h_output, dtype=np.float32)
                                    print(np.argmax(output))
                        dur_time = time.time() - start_time
                        print("### Detect Use %f mniutes" %(dur_time/60.))

def main():
    detect()

if __name__ == "__main__":
    main()    