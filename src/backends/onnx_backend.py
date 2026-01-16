import onnxruntime as ort
import cv2
import time
import platform

from .inference_backend import InferenceBackend

class OnnxBackend(InferenceBackend):
    def __init__(self, model_path, input_size=(640, 640), use_gpu=True):
        super().__init__(model_path, input_size)
        
        arch = platform.machine().lower()
        available_providers = ort.get_available_providers()
        providers = []
        provider_options = []
        options = None

        if use_gpu and 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            print(" GPU mode (CUDA) enabled.")
        
        elif "arm" in arch or "aarch64" in arch:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = True
            options.enable_mem_pattern = True
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers.append('CPUExecutionProvider')
        if not providers[0].startswith(('CUDA', 'XNNPACK')):
            print("Standard CPU mode selected.")

        print(f"Loading ONNX model: {model_path}...")
        
        self.session = ort.InferenceSession(
            str(model_path), 
            sess_options=options, 
            providers=providers,
            provider_options=provider_options if provider_options else None
        )
        
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