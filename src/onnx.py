from pathlib import Path

from .backends import OnnxBackend
from .pipeline import benchmark
from .helpers.utils import get_models_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_MODELS = PROJECT_ROOT / "models" / "onnx"

models_files = get_models_files(ONNX_MODELS, suffix=".onnx")

benchmark.bench(
    OnnxBackend, 
    models_files,
    PROJECT_ROOT / "results",
    output_file='benchmark_onnx.csv'
)