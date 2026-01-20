# SI5-Edge-AI — Benchmarking and Optimization of YOLO Models for Edge AI

This project aims to **benchmark, optimize, and evaluate YOLOv11 segmentation models** in an **Edge AI** context, with a specific application to a **person-following robot** based on computer vision.

The focus is on:
- **Accuracy** (COCO mAP for bounding boxes and masks),
- **Inference latency**,
- **Energy consumption**,

across multiple hardware platforms:
- PC (CPU / NVIDIA GPU),
- Raspberry Pi,
- Luxonis OAK-D Pro (Myriad X VPU).

---

## Project Objectives

- Evaluate the impact of various **model optimization techniques**:
  - quantization (FP32 / FP16 / INT8),
  - pruning (5% to 75%),
  - deployment formats (ONNX / BLOB).
- Compare performance across **multiple hardware platforms**.
- Analyze the **accuracy vs latency vs energy** trade-off.
- Provide a **reproducible benchmark pipeline** based on the COCO evaluation protocol.

---

## Models Used

- **YOLOv11-nano segmentation** (`yolo11n-seg`)
- Evaluated class: **person**
- Tasks:
  - object detection (bounding boxes),
  - instance segmentation (masks).

---

## Project Structure

```text
SI5-Edge-AI/
├── optimizations/
│   ├── pruning/                # Model pruning tools (model_optimizer)
│   │   └── model_optimizer/
│   │       ├── blob_convertor.py   # ONNX -> BLOB conversion for OAK-D
│   │       ├── calculate.py        # Optimization-related utilities
│   │       ├── prune.py            # Model pruning
│   │       └── ultralytics/        # Modified Ultralytics YOLO source
│   │
│   └── quantization/           # FP32 / FP16 / INT8 quantization
│       └── quantization.py
│
├── models/                     # YOLOv11 models
│   ├── yolo11n-seg.pt           # Base PyTorch model
│   ├── onnx/                    # ONNX models (PC / Raspberry Pi)
│   └── blob/                    # BLOB models (OAK-D)
│
├── results/                    # Benchmark results
│
├── src/                        # Main source code
│   ├── pipeline.py             # COCO benchmark pipeline
│   ├── onnx.py                 # ONNX Runtime benchmark
│   ├── oak_d.py                # OAK-D benchmark
│   ├── backends/               # Inference backends
│   ├── helpers/                # Utilities, config, YOLO post-processing
│   └── power_monitoring/       # Energy monitoring scripts
│
└── README.md
```

---

## Installation

This project uses [**uv**](https://docs.astral.sh/uv/) for fast and reliable Python dependency management.

### 1. Install uv

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

**For CPU-only inference:**
```powershell
uv sync
```

**For GPU inference and monitoring:**
```powershell
uv sync --group gpu
```

This installs `onnxruntime-gpu` and `pynvml` for NVIDIA GPU support.

---

## Usage

All scripts must be run with `uv run` to ensure the correct environment is used.

### Run Benchmarks

**ONNX Runtime (CPU/GPU):**  
Runs inference benchmarks using ONNX Runtime on ONNX models.
```powershell
uv run python -m src.onnx
```

**OAK-D:**  
Runs inference benchmarks on OAK-D hardware with BLOB models.
```powershell
uv run python -m src.oak_d
```

### Power Consumption Monitoring

**NVIDIA GPU Monitoring:**  
Monitors real-time power consumption of NVIDIA GPU during inference.
```powershell
uv run python .\src\power_monitoring\nvidia_gpu.py
```

**USB Power Meter Monitoring:**  
Extracts power consumption values (voltage, current, power) from a USB power meter display by analyzing video frames.
```powershell
uv run python .\src\power_monitoring\rasp.py
```
