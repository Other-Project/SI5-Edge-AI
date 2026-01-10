import depthai as dai
import json
import time
import numpy as np

from .inference_backend import InferenceBackend

class OakDBackend(InferenceBackend):
    def __init__(self, model_path, config_path, input_size=(640, 640)):
        super().__init__(model_path, input_size)
        
        with open(config_path, "r") as f:
            self.model_config = json.load(f)

        pipeline = dai.Pipeline()
        xin = pipeline.createXLinkIn()
        xin.setStreamName("in")
        
        nn = pipeline.createNeuralNetwork()
        nn.setBlobPath(model_path)

        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        
        xin.out.link(nn.input)
        
        xout = pipeline.createXLinkOut()
        xout.setStreamName("nn")
        nn.out.link(xout.input)

        self.device = dai.Device(pipeline)
        self.q_in = self.device.getInputQueue("in")
        self.q_nn = self.device.getOutputQueue("nn")

    def predict(self, image):
        input_img, scale, pad_x, pad_y = self.letterbox(image)
        
        frame = dai.ImgFrame()
        frame.setData(input_img.transpose(2, 0, 1).flatten())
        frame.setType(dai.ImgFrame.Type.BGR888p)
        frame.setWidth(self.input_size[0])
        frame.setHeight(self.input_size[1])
        
        t0 = time.perf_counter()
        self.q_in.send(frame)
        in_nn = self.q_nn.get()
        t_infer = (time.perf_counter() - t0) * 1000

        l0_data = np.array(in_nn.getLayerFp16("output0"))
        l1_data = np.array(in_nn.getLayerFp16("output1"))
        
        shape0 = self.model_config["shapes"]["output0"]
        shape1 = self.model_config["shapes"]["output1"]
        
        l0 = l0_data.reshape(1, shape0[1], shape0[2])
        l1 = l1_data.reshape(1, shape1[1], shape1[2], shape1[3])

        self.decoder.prepare_input_for_oakd(image.shape[:2], scale, pad_x, pad_y)
        self.decoder.segment_objects_from_oakd(l0, l1)

        return self.decoder, t_infer

    def close(self):
        if self.device:
            self.device.close()