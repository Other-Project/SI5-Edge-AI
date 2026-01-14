from .backends import OakDBackend
from .pipeline import BenchmarkPipeline
from .helpers.utils import get_dataset_paths
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, LABELS_FILE = get_dataset_paths("coco_person")
BLOB_MODELS = PROJECT_ROOT / "models" / "blob"
OAK_CONFIG = Path(__file__).parent / "helpers" / "config.json"
MAX_IMAGES = 100

benchmark = BenchmarkPipeline(DATA_DIR, LABELS_FILE)

models_files = {}

for model_path in BLOB_MODELS.iterdir():
    if model_path.suffix == ".blob":
        models_files[model_path.name] = model_path
        print(f"🔹 Modèle trouvé : {model_path.name}")

benchmark.bench(
    OakDBackend, 
    models_files, 
    PROJECT_ROOT / "results" / "oak_d", 
    max_images=MAX_IMAGES,
    backend_kwargs={"config_path": OAK_CONFIG}
)