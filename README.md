# CAN Anomaly Detector

A hybrid platform for anomaly detection on vehicle CAN bus data. The system architecture supports three independent detection layers — statistical analysis, SLTL formal verification, and LSTM-based deep learning — running in parallel. The LSTM predictor is the primary contribution of this project; the statistical and SLTL components provide the integration scaffolding and a baseline for comparison.

[![Language](https://img.shields.io/badge/C%2B%2B-45%25-blue?logo=cplusplus)](https://isocpp.org/)
[![Language](https://img.shields.io/badge/Python-38%25-yellow?logo=python)](https://python.org/)
[![Language](https://img.shields.io/badge/C-14%25-lightgrey?logo=c)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Build](https://img.shields.io/badge/build-QMake-green)](https://doc.qt.io/qt-6/qmake-manual.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](./LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Detection Layers](#detection-layers)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Input Formats](#input-formats)
- [Output](#output)
- [FAQ](#faq)
- [Author](#author)
- [License](#license)

---

## Overview

CAN Anomaly Detector reads raw CAN frames from live hardware interfaces, `candump` log files, Arduino serial proxies, or CSV datasets, and routes them through a multi-layer detection pipeline. Anomalies detected by any layer are logged to CSV with timestamps and metadata.

**Intended use cases:**

- Automotive cybersecurity research
- CAN bus intrusion detection
- Vehicle health monitoring
- Educational projects in embedded systems and time-series ML

---

## Detection Layers

| Layer | Method                                                                         | Status                |
| ----- | ------------------------------------------------------------------------------ | --------------------- |
| 1     | Statistical — Z-score / threshold-based outlier detection                     | Scaffolding           |
| 2     | SLTL — Signal Linear Temporal Logic formal property verification              | Scaffolding           |
| 3     | **LSTM — Deep learning sequence model for temporal anomaly prediction** | **Implemented** |

The LSTM predictor (`AnomalyPredictorLSTM`, `LSTMAnomaly.py`) is the primary implementation in this project. Layers 1 and 2 establish the pluggable multi-predictor interface and provide structural context; their logic is partial and serves as a baseline reference.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Data Receivers                        │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────────────┐ │
│  │ CANDump     │ │ Arduino      │ │ File / FileDir /       │ │
│  │ Receiver    │ │ Proxy        │ │ Streaming Receiver     │ │
│  └──────┬──────┘ └──────┬───────┘ └───────────┬────────────┘ │
└─────────┼───────────────┼────────────────────┼──────────────┘
          └───────────────┴────────────────────┘
                          │
               ┌──────────▼──────────┐
               │  AutoDetecting      │
               │  Receiver           │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │  CarDataProcessor   │
               │  Thread             │
               └──────────┬──────────┘
          ┌───────────────┼────────────────────┐
          ▼               ▼                    ▼
┌─────────────┐  ┌──────────────┐   ┌──────────────────┐
│  Statistical│  │    SLTL      │   │   LSTM Predictor │
│  Predictor  │  │  Predictor   │   │  (Python/ONNX)   │
│ (scaffolding│  │ (scaffolding)│   │  ← implemented   │
└──────┬──────┘  └──────┬───────┘   └────────┬─────────┘
       └────────────────┴──────────────────────┘
                          │
               ┌──────────▼──────────┐
               │   CSV Anomaly       │
               │   Logger            │
               └─────────────────────┘
```

The multithreaded architecture uses Qt's thread model — worker threads, processor threads, and fetcher threads — to decouple ingestion from prediction.

---

## Tech Stack

| Component                 | Technology                                      |
| ------------------------- | ----------------------------------------------- |
| Core application          | C++17                                           |
| Build system              | QMake (Qt Project)                              |
| Threading & I/O           | Qt 5/6 (`QThread`, `QMutex`, `QFile`)     |
| Formal verification layer | SLTL — custom C++ scaffolding                  |
| ML model training         | Python 3, TensorFlow / Keras (LSTM autoencoder) |
| ML model runtime          | C++ inference via loaded model or subprocess    |
| Hardware interface        | Arduino serial proxy (CAN-to-USB bridge)        |
| Log format                | `candump` (SocketCAN)                         |
| Output format             | CSV                                             |
| IDE support               | Qt Creator, Visual Studio, VS Code              |

---

## Project Structure

```
CAN-Anomaly-Detector/
│
├── main.cpp
│
├── # Data Receivers
├── datareceiver.{h,cpp}              # Abstract receiver interface
├── autodetectingreceiver.{h,cpp}     # Auto-selects receiver by source type
├── candumpreceiver.{h,cpp}           # candump log format reader
├── arduinoproxyreceiver.{h,cpp}      # Arduino serial-to-CAN bridge
├── filereceiver.{h,cpp}              # Single CSV/log file receiver
├── filedirreceiver.{h,cpp}           # Batch directory receiver
├── streamingreceiver.{h,cpp}         # Real-time streaming receiver
│
├── # Threading 
├── datareceiverthread.{h,cpp}        # Thread wrapper for data receiver
├── cardataprocessorthread.{h,cpp}    # CAN frame processing thread
├── datarowfetcherthread.{h,cpp}      # Row fetching worker thread
├── asyncpredictor.{h,cpp}            # Async anomaly predictor wrapper
├── predictworker.{h,cpp}             # Worker running predictor logic
├── lockers.{h,cpp}                   # Mutex / locking utilities
│
├── # Anomaly Predictors
├── anomalypredictor.{h,cpp}          # Abstract base class
├── anomalypredictorstatistic.{h,cpp} # Statistical predictor (scaffolding)
├── anomalypredictorsltl.{h,cpp}      # SLTL predictor (scaffolding)
├── anomalypredictorlstm.{h,cpp}      # LSTM predictor (implemented)
│
├── # Domain / Data Model
├── icardata.{h,cpp}                  # Abstract car data interface
├── mazda6cardata.{h,cpp}             # Mazda 6 CAN signal implementation
├── icansubscriber.h                  # Subscriber pattern interface
├── isltlproperty.{h,cpp}             # SLTL property interface
├── speedincreasesafterrpmincrease    # Concrete SLTL property: speed ↑ after RPM ↑
│   property.{h,cpp}             
│
├── # Output
├── csvanomalylogger.{h,cpp}          # Logs anomalies to CSV
├── anomalies.csv                     # Sample anomaly output
├── output.csv                        # Processed signal output
├── bad_rpm.csv                       # Labeled anomalous RPM dataset
├── ano.png                           # LSTM anomaly detection plot
│
├── # Python / ML
├── LSTMAnomaly.py                    # LSTM model training & evaluation
├── lstm/                             # Saved model weights and artifacts
├── stat/                             # Statistical baseline configs
│
├── anomaly_processor.pro             # QMake project file
└── anomaly_processor                 # Compiled binary (Linux)
```

---

## Installation

### Prerequisites

**C++ application:**

- Qt 5.12+ or Qt 6.x (with `qmake`)
- GCC / Clang with C++17 support
- Linux (native SocketCAN), or macOS/Windows with an Arduino CAN bridge

**Python LSTM trainer:**

- Python 3.8+
- TensorFlow 2.x / Keras
- NumPy, Pandas, Matplotlib, scikit-learn

### 1. Clone

```bash
git clone https://github.com/4lister/CAN-Anomaly-Detector.git
cd CAN-Anomaly-Detector
```

### 2. Build C++ application

```bash
qmake anomaly_processor.pro
make -j$(nproc)
```

### 3. Install Python dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### 4. Train LSTM model

Pre-trained artifacts are available in `lstm/`. To retrain:

```bash
python LSTMAnomaly.py
```

---

## Usage

```bash
# Run on a candump log file
./anomaly_processor --input path/to/candump.log

# Batch mode over a directory of logs
./anomaly_processor --dir path/to/logs/

# Live CAN via Arduino serial proxy
./anomaly_processor --serial /dev/ttyUSB0

# Train / evaluate LSTM model
python LSTMAnomaly.py
```

> Exact CLI flags depend on `main.cpp` configuration. Run `--help` if available.

---

## Input Formats

| Source                | Format                         | Class                    |
| --------------------- | ------------------------------ | ------------------------ |
| `candump` log files | SocketCAN text format          | `CANDumpReceiver`      |
| Arduino serial        | Raw serial frames over USB     | `ArduinoProxyReceiver` |
| CSV files             | Signal columns with timestamps | `FileReceiver`         |
| Directory of files    | Batch of any supported format  | `FileDirReceiver`      |
| Live stream           | Continuous real-time feed      | `StreamingReceiver`    |

**Example `candump` line:**

```
(1609459200.123456) vcan0 0C6#00000000DEADBEEF
```

**Example CSV row:**

```
timestamp,rpm,speed,throttle,...
1609459200,850,0,0,...
```

---

## Output

Detected anomalies are written to `anomalies.csv`:

```csv
timestamp,signal,value,predictor,severity
1609460012,rpm,9500,STATISTICAL,HIGH
1609460035,rpm,9800,LSTM,HIGH
```

The LSTM trainer produces a reconstruction error plot (`ano.png`) with a threshold line marking anomalous windows.

![Anomaly Detection Plot](./ano.png)

---

## FAQ

**What CAN hardware is supported?**
SocketCAN on Linux, Arduino-based CAN shields via serial proxy, and offline replay from `candump` log files. Any hardware producing `candump`-compatible output or raw serial CAN frames will work.

**Which vehicle models are supported?**
A concrete `Mazda6CarData` implementation is included. Other vehicles can be added by subclassing `ICarData` and mapping the relevant CAN IDs and signal decoding logic.

**How does the SLTL predictor work?**
SLTL (Signal Linear Temporal Logic) allows temporal properties of vehicle signals to be expressed as logical formulas — for example, *"speed must increase within N frames after RPM increases."* A concrete example is implemented in `SpeedIncreasesAfterRPMIncreasesProperty`. The SLTL layer is structural scaffolding; the LSTM predictor is the primary implemented detector.

**Do I need a real vehicle to test this?**
No. The included `bad_rpm.csv` and `output.csv` datasets, together with `candump` replay mode, support full offline testing.

**How is the LSTM model integrated into the C++ runtime?**
`AnomalyPredictorLSTM` handles inference in C++. The trained model (exported from `LSTMAnomaly.py`, stored in `lstm/`) is loaded at runtime. The inference mechanism — TensorFlow C API, ONNX, or subprocess — can be confirmed in `anomalypredictorlstm.cpp`.

**Can I add a new detection algorithm?**
Yes. Subclass `AnomalyPredictor` (see `anomalypredictor.h`), implement the required interface, and register the class in `main.cpp` or the processor thread setup.

**Why are `.vs`, `.vscode`, and `.qtc_clangd` folders committed?**
IDE configuration folders for Visual Studio, VS Code, and Qt Creator were not excluded in `.gitignore`. They can be safely deleted.

---

## Author

**4lister** — [@4lister](https://github.com/4lister)

---

## License

[MIT License](./LICENSE)
