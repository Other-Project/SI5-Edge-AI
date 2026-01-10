import cv2
import numpy as np

from .utils import draw_detections, nms, sigmoid, xywh2xyxy

PERSON_ID = 0

class Segment:
    def __init__(
        self,
        input_shape=[1, 3, 640, 640],
        input_height=640,
        input_width=640,
        conf_thres=0.7,
        iou_thres=0.5,
        num_masks=32,
    ):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        self.input_names = "images"
        self.input_shape = input_shape
        self.input_height = input_height
        self.input_width = input_width
        self.output_names = ["output0", "output1"]

    def segment_objects_from_oakd(self, output0, output1):

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(
            output0
        )
        self.mask_maps = self.process_mask_output(mask_pred, output1)

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input_for_oakd(self, original_shape, scale, pad_x, pad_y):
        self.orig_h, self.orig_w = original_shape
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        person_mask = class_ids == PERSON_ID

        box_predictions = box_predictions[person_mask]
        mask_predictions = mask_predictions[person_mask]
        scores = scores[person_mask]
        class_ids = class_ids[person_mask]

        if len(scores) == 0:
            return [], [], [], np.array([])

        self.boxes_net = box_predictions[:, :4].copy()
        boxes = self.extract_boxes(box_predictions)

        indices = nms(boxes, scores, self.iou_threshold)

        return (
            boxes[indices],
            scores[indices],
            class_ids[indices],
            mask_predictions[indices],
        )

    def process_mask_output(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)
        num_mask, mask_h, mask_w = mask_output.shape

        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_h, mask_w))

        ratio_h = mask_h / self.input_height 
        ratio_w = mask_w / self.input_width

        m_pad_y = int(self.pad_y * ratio_h)
        m_pad_x = int(self.pad_x * ratio_w)

        mask_maps = []

        for i in range(len(masks)):
            useful_mask = masks[i][m_pad_y : mask_h - m_pad_y, m_pad_x : mask_w - m_pad_x]
            
            full_mask = cv2.resize(useful_mask, (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR)
            
            full_mask = (full_mask > 0.5).astype(np.uint8)
            
            x1, y1, x2, y2 = map(int, self.boxes[i])
            final_mask = np.zeros((self.orig_h, self.orig_w), dtype=np.uint8)
            final_mask[y1:y2, x1:x2] = full_mask[y1:y2, x1:x2]
            
            mask_maps.append(final_mask)

        return np.array(mask_maps)

    def extract_boxes(self, box_predictions):
        boxes = box_predictions[:, :4]
        boxes = xywh2xyxy(boxes)

        boxes[:, [0, 2]] -= self.pad_x
        boxes[:, [1, 3]] -= self.pad_y

        boxes /= self.scale

        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.orig_h)

        return boxes


    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(
            image, self.boxes, self.scores, self.class_ids, mask_alpha
        )

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(
            image,
            self.boxes,
            self.scores,
            self.class_ids,
            mask_alpha,
            mask_maps=self.mask_maps,
        )

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
        )

        return boxes
