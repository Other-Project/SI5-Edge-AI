from abc import ABC, abstractmethod
import numpy as np
import cv2

from ..helpers.yolo_api import Segment

class InferenceBackend(ABC):
    def __init__(self, model_path, input_size=(640, 640), conf_thres=0.2, iou_thres=0.5):
        self.model_path = model_path
        self.input_size = input_size

        self.decoder = Segment(
            input_shape=[1, 3, input_size[1], input_size[0]],
            input_height=input_size[1],
            input_width=input_size[0],
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

    def letterbox(self, image):
        ih, iw = image.shape[:2]
        w, h = self.input_size
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        final_image = np.full((h, w, 3), 114, dtype=np.uint8)
        dx, dy = (w - nw) // 2, (h - nh) // 2
        final_image[dy:dy + nh, dx:dx + nw] = image_resized

        return final_image, scale, dx, dy

    @abstractmethod
    def predict(self, image):
        pass

    def close(self):
        pass