from pathlib import Path

from .backends import OakDBackend
from .pipeline import benchmark
from .helpers.utils import get_models_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOB_MODELS = PROJECT_ROOT / "models" / "blob"
OAK_CONFIG = Path(__file__).parent / "helpers" / "config.json"

models_files = get_models_files(BLOB_MODELS, suffix=".blob")

benchmark.bench(
    OakDBackend, 
    models_files, 
    PROJECT_ROOT / "results", 
    output_file='benchmark_oakd.csv',
    backend_kwargs={"config_path": OAK_CONFIG}
)