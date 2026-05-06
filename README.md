# 🚀 ESP32 Machine Learning HIL Benchmarking Pipeline

An end-to-end Hardware-in-the-Loop (HIL) benchmarking system for evaluating dynamic Machine Learning architectures (Dense, Conv2D, LSTM) on an ESP32 Microcontroller. 

This project bridges **PyTorch** and **TensorFlow/Keras** ecosystems, automatically converting dynamic architectures across ONNX and TensorFlow Lite Micro formats. It conducts automated UART-based memory/latency stress tests on actual ESP32 hardware and visualizes the results (Inference Time, RAM Usage, MSE, and Architecture Graphs) via a local Plotly Dash dashboard.

---

## 🏗️ System Architecture

1. **Dynamic Training**: N-layer modifiable architectures trained in PyTorch and Keras.
2. **Pipeline Conversion**: `PyTorch -> ONNX -> TFLite` & `Keras -> TFLite`.
3. **Data Extraction**: Automated parsing of Keras/PyTorch execution graphs and parameters.
4. **Hardware Benchmarking**: 2-Step Handshake UART protocol dynamically allocates memory, runs inference, and captures metrics on the ESP32.
5. **Analytics Dashboard**: Interactive UI to compare int8 vs float32, Batch Normalization impact, and isolate cascading hardware failures (Flaky models).

---

## 💻 Environment Setup (Conda)

Due to conflicting dependencies between PyTorch, TensorFlow, and ONNX conversion tools, this project strictly relies on **two separate Conda environments**.

### 1. PyTorch Environment (`torch_env`)
Used for training PyTorch models and exporting them to ONNX.
```bash
conda create -n torch_env python=3.10
conda activate torch_env
pip install torch torchvision torchaudio pandas numpy
```

### 2. TensorFlow Environment (`tf_env`)
Used for Keras training, ONNX-to-TFLite conversion, Hardware Benchmarking, and the Dashboard.
```bash
conda create -n tf_env python=3.10
conda activate tf_env
pip install tensorflow pandas numpy pyserial dash plotly onnx2tf
```

---

## 🔄 Step-by-Step Workflow

⚠️ **Crucial Naming Convention:**
All models must follow the schema: `{arch}_{layers}_{seq}_{bn}` (e.g., `lstm_3_10_OFF`).

### Phase 1: Train Models & Extract Metadata
**In your PyTorch Environment (`torch_env`):**
1. Run `python torch_exp_runner.py` to train permutations of Dense/Conv/LSTM models.
2. Run `python export_onnx.py` to convert `.pt` files to `.onnx`.
3. Run `python extract_torch_info.py` to generate `torch_structures.json`.

**In your TensorFlow Environment (`tf_env`):**
1. Run your Keras training script to generate `.keras` models.
2. Run `python extract_keras_info.py` to generate `keras_structures.json`.

### Phase 2: Convert to TFLite
**Stay in your TensorFlow Environment (`tf_env`):**
1. Run `python onnx_to_tflite.py`. This reads your ONNX models, dynamically infers shapes, runs the TFLite integer calibrator, and outputs `torch_*.tflite` files.
2. Run your Keras TFLite conversion (if not already done during training).
3. Run `python extract_tflite_info.py` to map the execution graphs of all generated `.tflite` files into `tflite_structures.json`.

### Phase 3: Hardware-in-the-Loop Benchmarking
1. Flash `main.cpp` to your ESP32 (Ensure the partition table allocates enough space for your app).
2. Connect your ESP32 via USB.
3. Open `benchmark.py` and ensure `SERIAL_PORT` matches your device (e.g., `/dev/tty.usbserial-0001` or `COM3`).
4. Run `python benchmark.py`.
   * *The script will randomize model order, upload binaries via UART, execute IMU test sequences, calculate MSE, and save a CSV report to `/benchmarks`.*
   * *Tip: Run this script 3-4 times to generate multiple reports for Stability Analysis!*

### Phase 4: Data Analytics
1. Run `python dashboard.py`.
2. Open your web browser to `http://127.0.0.1:8050`.
3. Use the dashboard to:
   * Compare **PyTorch vs Keras** inference speeds.
   * Visualize the massive memory spike caused by unrolled LSTMs.
   * Analyze the penalty of keeping Batch Normalization `ON`.
   * Click any graph bar to inspect the Keras parameters and TFLite execution graphs.

---

## 🛑 Known Limitations & Troubleshooting

* **ESP32 Malloc Fails (`ERR_SIZE`):** The ESP32 has limited heap RAM. Deep, unrolled LSTM models at high sequence lengths will exceed ~120KB of Arena memory and fail to allocate. The dashboard isolates these failures gracefully.
* **Hybrid Quantization Failures:** TensorFlow Lite Micro has limited operational support for Hybrid quantization. Use `int8` (Full Integer) or `float32` for microcontrollers.
* **Cascading Flaky Models:** Sometimes a TFLite model crashes mid-execution and corrupts the ESP32 heap, causing the *next* valid model to fail. The dashboard's **Stability Analysis Tab** identifies this by cross-referencing multiple benchmark runs.
