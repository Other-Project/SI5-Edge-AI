import os
from backends import OnnxBackend, OakDBackend
from pipeline import BenchmarkPipeline

DATA_DIR = "data"
LABELS_FILE = "labels.json"
OAK_CONFIG = "helpers/config.json"

MODEL_ONNX = "models/yolo11n-seg.onnx"
MODEL_BLOB = "models/yolo11n-seg_openvino_2022.1_6shave.blob"

MAX_IMAGES = 100  

def display_report(name, stats):
    print(f"\n{'='*10} RESULTS : {name} {'='*10}")
    
    print(f"INFERENCE TIME :")
    print(f"   • Average    : {stats['time_avg']:.2f} ms")
    print(f"   • Minimum    : {stats['time_min']:.2f} ms")
    print(f"   • Maximum    : {stats['time_max']:.2f} ms")
    print(f"   • Stability  : ±{stats['time_std']:.2f} ms (Standard Deviation)")
    
    print(f"ACCURACY :")
    print(f"   • mAP Box    : {stats['mAP_box']:.4f}")
    print(f"   • mAP Mask   : {stats['mAP_mask']:.4f}")
    print("="*40 + "\n")

benchmark = BenchmarkPipeline(DATA_DIR, LABELS_FILE)

backend_pc = OnnxBackend(MODEL_ONNX)

stats_pc = benchmark.run(backend_pc, max_images=MAX_IMAGES)

display_report("PC (ONNX Runtime)", stats_pc)

backend_pc.close()

backend_oak = OakDBackend(MODEL_BLOB, OAK_CONFIG)

stats_oak = benchmark.run(backend_oak, max_images=MAX_IMAGES)
display_report("OAK-D (Myriad X)", stats_oak)

backend_oak.close()
