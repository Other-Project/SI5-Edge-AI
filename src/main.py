from .backends import OnnxBackend, OakDBackend
from .pipeline import BenchmarkPipeline
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LABELS_FILE = DATA_DIR / "labels.json"

MODEL_ONNX = PROJECT_ROOT / "models" / "yolo11n-seg.onnx"
MODEL_BLOB = PROJECT_ROOT / "models" / "yolo11n-seg_openvino_2022.1_6shave.blob"

OAK_CONFIG = Path(__file__).parent / "helpers" / "config.json"

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
