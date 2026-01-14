import os
from pathlib import Path
import onnxruntime as ort

from .backends import OnnxBackend
from .pipeline import BenchmarkPipeline
from .helpers.utils import get_dataset_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, LABELS_FILE = get_dataset_paths("coco_person")
ONNX_MODELS = PROJECT_ROOT / "models"
MAX_IMAGES = 100

benchmark = BenchmarkPipeline(DATA_DIR, LABELS_FILE)

models_files = {}

for model_path in ONNX_MODELS.iterdir():
    if model_path.suffix == ".onnx":
        models_files[model_path.name] = model_path
        print(f"🔹 Modèle trouvé : {model_path.name}")

benchmark.bench(OnnxBackend, models_files, PROJECT_ROOT / "results", max_images=MAX_IMAGES)