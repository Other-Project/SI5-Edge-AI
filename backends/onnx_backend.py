import onnxruntime as ort
import cv2
import time

from .inference_backend import InferenceBackend

class OnnxBackend(InferenceBackend):
    def __init__(self, model_path, input_size=(640, 640), use_gpu=False):
        super().__init__(model_path, input_size)

        providers = ['CPUExecutionProvider']
        if use_gpu:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"GPU Mode enabled")
            else:
                print("GPU requested but 'CUDAExecutionProvider' not found. Falling back to CPU.")
        else:
            print("CPU Mode selected.")

        print(f"Loading ONNX model: {model_path}...")
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, image):
        input_img, scale, pad_x, pad_y = self.letterbox(image)

        blob = cv2.dnn.blobFromImage(input_img, 1/255.0, (0,0), swapRB=True, crop=False)
        
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        t_infer = (time.perf_counter() - t0) * 1000

        self.decoder.prepare_input_for_oakd(image.shape[:2], scale, pad_x, pad_y)
        self.decoder.segment_objects_from_oakd(outputs[0], outputs[1])
        
        return self.decoder, t_infer