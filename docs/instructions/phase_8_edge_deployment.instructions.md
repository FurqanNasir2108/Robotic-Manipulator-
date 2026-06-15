# Phase 8: Edge Deployment and IoT Integration

> **Prerequisite:** Phases 6–7 complete. Read `00_main_orchestrator.instructions.md` first.

---

## Goal

Study whether trained trajectory generation models can be deployed for real-time or near-real-time use on edge devices. Export, optimize, benchmark, and simulate a sensor-to-trajectory inference pipeline.

---

## Relationship to Overall Workflow

```
Phase 7 (Evaluation) → Phase 8 (Edge Deployment) → Phase 9 (Documentation)
                            ↑
        Uses trajectory analysis results (Phase 6)
        and evaluation comparison (Phase 7) to select
        the best model for deployment via Pareto analysis
```

Edge deployment comes **after** trajectory analysis and evaluation. The deployment model is selected based on the accuracy-latency-size Pareto front computed from Phase 6 and Phase 7 results.

---

## System Architecture

```
Task condition / IoT sensor input
        ↓
Edge device (CPU / Jetson / RPi)
        ↓
Optimized model (ONNX / TensorRT)
        ↓
Generated joint trajectory q1(t), q2(t), q3(t)
        ↓
Robot controller / simulator
        ↓
Trajectory feedback and logging
```

---

## Supported Models

| Model | Deployment Priority | Export Complexity | Notes |
|-------|-------------------|-------------------|-------|
| cVAE | **Primary** | Medium | One-shot inference; export `decode(z, condition)` |
| CNN | Secondary | Low | Pure feedforward |
| CNN+LSTM | Secondary | Low-Medium | Hidden state management |
| CNN+GRU | Secondary | Low-Medium | Hidden state management |
| Diffusion (DDIM) | Optional | High | Export denoiser; sampling loop in Python |

### Export Strategy for Generative Models

**cVAE:** Cannot export `generate()` directly due to `torch.randn` (stochastic op). Export `decode(z, condition)` as ONNX. Sample `z ~ N(0, I)` in Python before calling ONNX.

**Diffusion:** Cannot export iterative sampling loop as single ONNX graph. Export `denoise_step(x_t, t, condition)` as ONNX. Implement DDIM loop in Python calling ONNX per step.

---

## Supported Hardware Targets

| Device | Compute | Runtime | Priority |
|--------|---------|---------|----------|
| CPU-only laptop | x86/ARM CPU | ONNX Runtime | Required (development fallback) |
| NVIDIA Jetson Nano | GPU (128 CUDA) | TensorRT / ONNX Runtime | Optional |
| NVIDIA Jetson Orin Nano | GPU (1024 CUDA) | TensorRT / ONNX Runtime | Optional |
| Raspberry Pi 4/5 | ARM CPU | ONNX Runtime | Optional |
| Coral TPU | TPU | TFLite | Optional (if compatible) |

**Rule:** Do not require physical edge hardware for first implementation. Provide CPU-only fallback. All device-specific code must be isolated.

---

## Tasks

### 8.1 Model Export Pipeline

Implement in `src/edge_deployment/model_export.py` and `onnx_export.py`:

```python
def export_to_torchscript(model, example_input, output_path)
def export_to_onnx(model, example_input, output_path, opset_version=17)
def validate_onnx_export(pytorch_model, onnx_path, test_input, rtol=1e-5)
def export_cvae_decoder(cvae_model, example_z, example_condition, output_path)
def export_diffusion_denoiser(diffusion_model, example_inputs, output_path)
```

**Pipeline:**
1. Load trained PyTorch model
2. Export model to ONNX
3. Validate ONNX output against PyTorch output (joint RMSE < 0.001)
4. Save exported model to `results/checkpoints/onnx/`

### 8.2 Model Optimization

Implement in `src/edge_deployment/optimization_utils.py`:

```python
def quantize_dynamic(onnx_path, output_path, weight_type='int8')
def quantize_static(onnx_path, output_path, calibration_data)
def optimize_onnx_graph(onnx_path, output_path, optimization_level='all')
def convert_to_fp16(onnx_path, output_path)
```

Quantization levels: FP32 → FP16 → INT8 (dynamic) → INT8 (static with calibration).

### 8.3 Runtime Inference

Implement in `src/edge_deployment/runtime_inference.py`:

```python
class ONNXInferenceSession:
    def __init__(self, onnx_path, device='cpu')
    def predict(self, condition) -> np.ndarray
    def predict_cvae(self, z, condition) -> np.ndarray
    def predict_diffusion_step(self, x_t, t, condition) -> np.ndarray

class TensorRTInferenceSession:
    def __init__(self, engine_path, device='cuda')
    def predict(self, condition) -> np.ndarray
```

### 8.4 Edge Benchmarking

Implement in `src/edge_deployment/edge_benchmark.py`:

```python
def benchmark_latency(session, test_inputs, n_runs=1000, warmup=100) -> dict
    # Returns: mean, median, P50, P95, P99 latency in ms

def benchmark_throughput(session, test_inputs, duration_sec=10) -> float
    # Returns: trajectories per second

def benchmark_memory(session, test_input) -> dict
    # Returns: peak RAM, peak GPU memory (if available)

def model_size_mb(model_path) -> float

def pareto_analysis(benchmark_results, accuracy_results) -> dict
    # Returns: Pareto-optimal models on accuracy-latency-size front
```

### 8.5 Device Monitoring

Implement in `src/edge_deployment/device_monitor.py`:

```python
class DeviceMonitor:
    def start()
    def stop()
    def get_cpu_usage() -> float
    def get_gpu_usage() -> float  # if available
    def get_ram_usage() -> float
    def get_gpu_memory() -> float  # if available
    def get_power_estimate() -> float  # if available
    def get_summary() -> dict
```

### 8.6 Simulated Sensor Interface

Implement in `src/edge_deployment/sensor_interface.py`:

```python
class SimulatedSensorStream:
    def __init__(self, test_dataset, rate_hz=10)
    def get_next_condition() -> dict  # goal pose, waypoints, shape params
    def reset()

class SensorPreprocessor:
    def __init__(self, normalizer)
    def preprocess(raw_condition) -> np.ndarray
```

Future hooks for real sensors (encoder, IMU) are defined as abstract interfaces.

### 8.7 Communication Layer

Implement in `src/edge_deployment/communication.py`:

```python
class RESTServer:
    def __init__(self, inference_session, host, port)
    def start()
    # POST /predict: condition → trajectory

class MQTTClient:
    def __init__(self, broker, topic, inference_session)
    def start()
    # Subscribe to condition topic, publish trajectory response
```

Optional — only implement REST as primary. MQTT, WebSocket, ROS2 as future extensions.

### 8.8 Controller Interface

Implement in `src/edge_deployment/controller_interface.py`:

```python
class SimulatorController:
    def __init__(self, manipulator)
    def execute_trajectory(trajectory) -> dict  # execution log
    def validate_execution(trajectory, executed) -> dict
```

### 8.9 Real-Time Inference Loop

Implement in `scripts/run_edge_inference_loop.py`:

```python
while system_is_running:
    condition = sensor.get_next_condition()
    processed = preprocessor.preprocess(condition)
    trajectory = session.predict(processed)
    feasible = feasibility_check(trajectory)
    if feasible:
        controller.execute_trajectory(trajectory)
    monitor.log(latency, quality, feasibility)
```

---

## Edge-Specific Evaluation Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| Average inference latency | ms | Mean prediction time |
| P50 / P95 / P99 latency | ms | Latency percentiles |
| Model size | MB | File size of exported model |
| Peak RAM usage | MB | Maximum memory during inference |
| Peak GPU memory | MB | Maximum GPU memory (if applicable) |
| CPU utilization | % | During inference |
| GPU utilization | % | During inference (if applicable) |
| Throughput | traj/sec | Trajectories generated per second |
| Energy per inference | mJ | Estimated energy (if power monitoring available) |
| PyTorch vs ONNX RMSE | rad | Numerical deviation between original and exported model |
| Task-space error after deployment | m | EE position error of deployed model |
| Feasibility violation rate | % | Constraint violations of deployed model output |
| Smoothness degradation | % | Smoothness metric change after quantization |
| Real-time success rate | % | Fraction of inferences completing within deadline |

---

## Phase-Wise Sub-Tasks

### Sub-Phase 8.1: Requirements and Config
- **Goal:** Define deployment requirements, create config
- **Tasks:** Create `configs/edge_deployment.yaml`, implement `src/edge_deployment/config.py`
- **Inputs:** Trained model paths, evaluation results
- **Outputs:** Config file, config loader
- **Deliverables:** `configs/edge_deployment.yaml`
- **Completion check:** Config loads without errors

### Sub-Phase 8.2: Model Export
- **Goal:** Export PyTorch models to ONNX
- **Tasks:** Implement `model_export.py`, `onnx_export.py`; export cVAE, CNN, and optionally diffusion
- **Inputs:** Trained PyTorch checkpoints
- **Outputs:** `.onnx` files in `results/checkpoints/onnx/`
- **Deliverables:** Export script, ONNX models
- **Completion check:** ONNX files created, `onnx.checker.check_model()` passes

### Sub-Phase 8.3: Export Validation
- **Goal:** Verify numerical accuracy of exported models
- **Tasks:** Compare PyTorch vs ONNX output on test set
- **Inputs:** PyTorch model, ONNX model, test data
- **Outputs:** Joint RMSE deviation report
- **Deliverables:** `scripts/compare_pytorch_vs_onnx.py`, deviation results
- **Completion check:** Joint RMSE between PyTorch and ONNX < 0.001

### Sub-Phase 8.4: ONNX Runtime Inference
- **Goal:** Implement ONNX Runtime inference wrapper
- **Tasks:** Implement `runtime_inference.py`
- **Inputs:** ONNX model path
- **Outputs:** Unified inference API
- **Deliverables:** `ONNXInferenceSession` class
- **Completion check:** Can generate trajectories from ONNX model

### Sub-Phase 8.5: Quantization and Optimization
- **Goal:** Apply quantization and graph optimization
- **Tasks:** Implement `optimization_utils.py`; quantize to FP16, INT8
- **Inputs:** FP32 ONNX model
- **Outputs:** Quantized ONNX models
- **Deliverables:** FP16 and INT8 model variants
- **Completion check:** Quantized models produce valid trajectories; accuracy degradation < 5%

### Sub-Phase 8.6: Benchmarking
- **Goal:** Measure latency, throughput, memory, model size
- **Tasks:** Implement `edge_benchmark.py`, `device_monitor.py`
- **Inputs:** All exported model variants
- **Outputs:** Benchmark results JSON/CSV
- **Deliverables:** `scripts/benchmark_edge_inference.py`, `scripts/profile_memory_latency.py`
- **Completion check:** Benchmark results for all model × quantization combinations

### Sub-Phase 8.7: Simulated IoT Sensor Input
- **Goal:** Implement simulated sensor stream
- **Tasks:** Implement `sensor_interface.py`
- **Inputs:** Test dataset
- **Outputs:** Streaming condition iterator
- **Deliverables:** `scripts/test_sensor_stream.py`
- **Completion check:** Sensor stream produces valid conditions at configured rate

### Sub-Phase 8.8: Real-Time Inference Loop
- **Goal:** Implement end-to-end sensor-to-trajectory pipeline
- **Tasks:** Implement `controller_interface.py`, `scripts/run_edge_inference_loop.py`
- **Inputs:** Sensor stream, ONNX model, simulator
- **Outputs:** Per-iteration latency/quality log
- **Deliverables:** Real-time loop script, execution log
- **Completion check:** Loop runs for 100+ iterations with logged metrics

### Sub-Phase 8.9: Integration with Trajectory Analysis
- **Goal:** Validate deployed model trajectory quality
- **Tasks:** Run trajectory analysis metrics on ONNX model output
- **Inputs:** ONNX-generated trajectories, trajectory analysis module
- **Outputs:** Quality comparison: PyTorch vs ONNX vs quantized
- **Deliverables:** Deployment quality report
- **Completion check:** Deployed model maintains feasibility rate >95%

### Sub-Phase 8.10: Deployment Report
- **Goal:** Generate thesis/paper-ready deployment report
- **Tasks:** Pareto analysis, comparison figures, summary tables
- **Inputs:** All benchmark and quality results
- **Outputs:** Figures, JSON, MD report
- **Deliverables:** `figures/edge_deployment/`, `results/metrics/edge_deployment/deployment_summary.md`
- **Completion check:** Report contains latency, accuracy, Pareto front, model size comparison

---

## TensorRT Export (Optional, NVIDIA Only)

Implement in `src/edge_deployment/tensorrt_export.py`:

```python
def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16=True)
def validate_tensorrt_engine(engine_path, test_input)
```

Guard all TensorRT code with:
```python
try:
    import tensorrt as trt
except ImportError:
    trt = None
    warnings.warn("TensorRT not available; skipping TensorRT export")
```

---

## Communication Protocols (Future Extensions)

| Protocol | Use Case | Priority |
|----------|----------|----------|
| REST (Flask) | Development, testing | Primary |
| MQTT | IoT sensor integration | Optional |
| WebSocket | Real-time bidirectional | Optional |
| ROS2 topics | Robotics middleware | Future |
| Serial (UART) | Microcontroller bridge | Future |

---

## Future Hardware Integration (Not Implemented Now)

- Encoder feedback for closed-loop control
- IMU feedback for state estimation
- Motor driver interface for direct actuation
- Microcontroller bridge (ESP32 / Arduino) for low-level control
- Edge device as dedicated inference unit in a robot control stack

---

## Risks and Mitigation

| Risk | Mitigation |
|------|-----------|
| ONNX export fails for cVAE | Export `decode()` only; sample `z` in Python |
| ONNX export fails for Diffusion | Export denoiser only; DDIM loop in Python |
| No GPU available | CPU-only ONNX Runtime fallback (always supported) |
| TensorRT not installed | Skip TensorRT; use ONNX Runtime |
| Quantization degrades accuracy severely | Report degradation; use FP16 instead of INT8 |
| No physical edge hardware | CPU benchmarks + simulated constraints; document as "simulated edge" |
| MQTT broker not available | REST-only fallback |

---

## Strict Rules

- Do not optimize a model before validating baseline PyTorch output.
- Do not claim real-time performance without measuring latency.
- Always compare optimized output with original model output numerically.
- Keep VAE as the first deployment target.
- Treat diffusion as optional or accelerated deployment.
- Do not require physical hardware for the first implementation.
- Provide CPU-only fallback for all benchmarks.
- Keep all device-specific code isolated behind config flags.
- Use config files for device selection — do not hardcode device names.
- Save benchmark results as CSV/JSON.
- Save deployment plots in `figures/edge_deployment/`.
- Add clear error handling for missing ONNX Runtime, TensorRT, CUDA, or hardware sensors.
- Do not ignore trajectory quality during deployment — always validate.

---

## Deliverables

- [ ] `src/edge_deployment/` module (11 files)
- [ ] `configs/edge_deployment.yaml`
- [ ] `scripts/export_model_to_onnx.py`
- [ ] `scripts/benchmark_edge_inference.py`
- [ ] `scripts/run_edge_inference_loop.py`
- [ ] `scripts/test_sensor_stream.py`
- [ ] `scripts/compare_pytorch_vs_onnx.py`
- [ ] `scripts/profile_memory_latency.py`
- [ ] ONNX models in `results/checkpoints/onnx/`
- [ ] Benchmark results in `results/metrics/edge_deployment/`
- [ ] Figures in `figures/edge_deployment/`
- [ ] Deployment summary report
- [ ] `tests/unit/test_edge_deployment.py`
- [ ] `tests/integration/test_edge_pipeline.py`
- [ ] `tests/integration/test_realtime_loop.py`

---

## Go/No-Go Checks

| Check | Action if Failed |
|-------|-----------------|
| ONNX export produces invalid graph | Check unsupported ops; simplify model forward pass |
| PyTorch vs ONNX RMSE > 0.001 | Debug export; check input/output shapes; verify opset version |
| Quantized model accuracy drops >10% | Use FP16 instead of INT8; report both |
| Latency exceeds real-time deadline | Try smaller model; reduce DDIM steps; report limitation |
| Memory exceeds device constraints | Reduce batch size to 1; try quantization; report limitation |
| Sensor stream desynchronizes | Add timeout and retry logic; log dropped conditions |

---
