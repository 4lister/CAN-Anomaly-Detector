# 🚗 CAN Anomaly Detector

> A hybrid software platform for real-time anomaly detection on vehicle CAN bus data — combining statistical analysis, SLTL formal verification, and deep LSTM neural networks.

[![Language](https://img.shields.io/badge/C%2B%2B-45%25-blue?logo=cplusplus)](https://isocpp.org/)
[![Language](https://img.shields.io/badge/Python-38%25-yellow?logo=python)](https://python.org/)
[![Language](https://img.shields.io/badge/C-14%25-lightgrey?logo=c)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Build](https://img.shields.io/badge/build-QMake-green)](https://doc.qt.io/qt-6/qmake-manual.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](./LICENSE)

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Build](#-installation--build)
- [Usage](#-usage)
- [Data Sources & Input Formats](#-data-sources--input-formats)
- [Example Output](#-example-output)
- [FAQ](#-faq)
- [Author](#-author)
- [License](#-license)

---

## 📖 About

**CAN Anomaly Detector** is a multi-layer anomaly detection platform for automotive CAN bus data. It reads raw CAN frames — from live hardware interfaces, `candump` log files, Arduino serial proxies, or CSV datasets — and applies three independent detection strategies in parallel:

| Layer | Method | Description |
|---|---|---|
| 1 | **Statistical** | Z-score / threshold-based outlier detection on signal values |
| 2 | **SLTL** | Signal Linear Temporal Logic formal property verification |
| 3 | **LSTM** | Deep learning sequence model for temporal anomaly prediction |

This makes the system suitable for automotive cybersecurity research, vehicle health monitoring, CAN intrusion detection, and educational projects in embedded systems and machine learning.

**Who is it for?**
- Automotive security researchers
- Embedded systems engineers
- Data scientists working with vehicle telemetry
- Students studying CAN bus protocols, formal methods, or time-series ML

---

## ✨ Features

- 🔌 **Multiple data sources** — live CAN interfaces, `candump` log files, Arduino serial proxy, CSV files, and directory batch processing
- 🧠 **Three-layer detection** — statistical, formal (SLTL), and LSTM-based anomaly predictors running in parallel
- ⚡ **Asynchronous processing** — multithreaded architecture using Qt's thread model (worker threads, processor threads, fetcher threads)
- 📝 **CSV anomaly logging** — detected anomalies are automatically exported to `anomalies.csv` with timestamps and metadata
- 🚙 **Vehicle-specific data model** — includes a concrete `Mazda6CarData` implementation with real CAN signal mappings (RPM, speed, etc.)
- 🔁 **Streaming & batch modes** — supports real-time streaming receiver as well as offline file/directory processing
- 🐍 **Python LSTM trainer** — `LSTMAnomaly.py` trains and evaluates the neural network model on captured CAN data
- 📊 **Pre-labeled datasets** — includes `bad_rpm.csv` and `output.csv` for immediate testing and model training
- 🔧 **Pluggable predictor interface** — new detection algorithms can be added by implementing the `AnomalyPredictor` base class
- 🖼️ **Anomaly visualization** — `ano.png` shows an example anomaly detection plot from the LSTM model

---

## 🏗️ Architecture Overview

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
│  Predictor  │  │  Predictor   │   │  (via Python /   │
│             │  │              │   │   ONNX model)    │
└──────┬──────┘  └──────┬───────┘   └────────┬─────────┘
       └────────────────┴──────────────────────┘
                          │
               ┌──────────▼──────────┐
               │   CSV Anomaly       │
               │   Logger            │
               └─────────────────────┘
```

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Core application | C++17 |
| Build system | QMake (Qt Project) |
| Threading & I/O | Qt 5/6 (`QThread`, `QMutex`, `QFile`) |
| Formal verification | SLTL (Signal Linear Temporal Logic) — custom C++ impl |
| ML model training | Python 3, TensorFlow / Keras (LSTM) |
| ML model runtime | C++ inference (via loaded model or subprocess) |
| Hardware interface | Arduino serial proxy (CAN-to-USB bridge) |
| Log format | `candump` (SocketCAN) |
| Output format | CSV |
| IDE support | Qt Creator (`.qtc_clangd`), Visual Studio (`.vs`), VS Code (`.vscode`) |

---

## 📁 Project Structure

```
CAN-Anomaly-Detector/
│
├── main.cpp                          # Entry point
│
├── # ── Data Receivers ───────────────────────────────────────
├── datareceiver.{h,cpp}              # Abstract data receiver interface
├── autodetectingreceiver.{h,cpp}     # Auto-selects receiver by source type
├── candumpreceiver.{h,cpp}           # Reads candump log format
├── arduinoproxyreceiver.{h,cpp}      # Arduino serial-to-CAN bridge receiver
├── filereceiver.{h,cpp}              # Single CSV/log file receiver
├── filedirreceiver.{h,cpp}           # Batch directory receiver
├── streamingreceiver.{h,cpp}         # Real-time streaming receiver
│
├── # ── Threading ────────────────────────────────────────────
├── datareceiverthread.{h,cpp}        # Thread wrapper for data receiver
├── cardataprocessorthread.{h,cpp}    # CAN frame processing thread
├── datarowfetcherthread.{h,cpp}      # Row fetching worker thread
├── asyncpredictor.{h,cpp}            # Async anomaly predictor wrapper
├── predictworker.{h,cpp}             # Worker running predictor logic
├── lockers.{h,cpp}                   # Mutex / locking utilities
│
├── # ── Anomaly Predictors ───────────────────────────────────
├── anomalypredictor.{h,cpp}          # Abstract predictor base class
├── anomalypredictorstatistic.{h,cpp} # Statistical (threshold) predictor
├── anomalypredictorsltl.{h,cpp}      # SLTL formal verification predictor
├── anomalypredictorlstm.{h,cpp}      # LSTM deep learning predictor
│
├── # ── Domain / Data Model ──────────────────────────────────
├── icardata.{h,cpp}                  # Abstract car data interface
├── mazda6cardata.{h,cpp}             # Mazda 6 CAN signal implementation
├── icansubscriber.h                  # Subscriber pattern interface
├── isltlproperty.{h,cpp}             # SLTL property interface
├── speedincreasesafterrpmincrease    # Concrete SLTL property: speed ↑ after RPM ↑
│   property.{h,cpp}
│
├── # ── Output ───────────────────────────────────────────────
├── csvanomalylogger.{h,cpp}          # Logs anomalies to CSV
├── anomalies.csv                     # Sample anomaly output
├── output.csv                        # Processed signal output
├── bad_rpm.csv                       # Labeled anomalous RPM data
├── ano.png                           # Example anomaly visualization
│
├── # ── Python / ML ──────────────────────────────────────────
├── LSTMAnomaly.py                    # LSTM model training & evaluation
├── lstm/                             # LSTM model artifacts / saved weights
├── stat/                             # Statistical baseline data / configs
│
├── # ── Build ────────────────────────────────────────────────
├── anomaly_processor.pro             # QMake project file
├── anomaly_processor                 # Compiled binary (Linux)
│
└── .gitignore
```

---

## ⚙️ Installation & Build

### Prerequisites

**C++ application:**
- Qt 5.12+ or Qt 6.x (with `qmake`)
- GCC / Clang with C++17 support
- Linux (native SocketCAN) or macOS/Windows with an Arduino CAN bridge

**Python LSTM trainer:**
- Python 3.8+
- TensorFlow 2.x / Keras
- NumPy, Pandas, Matplotlib

---

### 1. Clone the repository

```bash
git clone https://github.com/4lister/CAN-Anomaly-Detector.git
cd CAN-Anomaly-Detector
```

### 2. Build the C++ application

```bash
# Using Qt Creator: open anomaly_processor.pro and click Build
# Or from the command line:
qmake anomaly_processor.pro
make -j$(nproc)
```

The compiled binary `anomaly_processor` will appear in the project root.

### 3. Install Python dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### 4. Train the LSTM model (optional, pre-trained artifacts may be in `lstm/`)

```bash
python LSTMAnomaly.py
```

---

## 🚀 Usage

### Run on a `candump` log file

```bash
./anomaly_processor --input path/to/candump.log
```

### Run on a directory of CAN logs (batch mode)

```bash
./anomaly_processor --dir path/to/logs/
```

### Run with Arduino serial proxy (live CAN)

Connect your Arduino CAN shield and specify the serial port:

```bash
./anomaly_processor --serial /dev/ttyUSB0
```

### Run the Python LSTM trainer

```bash
python LSTMAnomaly.py
```

The script reads CAN signal data from CSV, trains an LSTM autoencoder, plots reconstruction error, and exports anomaly predictions.

> **Note:** Exact CLI flags depend on `main.cpp` configuration. Check the source or use `--help` if available.

---

## 📂 Data Sources & Input Formats

| Source | Format | Class |
|---|---|---|
| `candump` log files | SocketCAN text format | `CANDumpReceiver` |
| Arduino serial | Raw serial frames over USB | `ArduinoProxyReceiver` |
| CSV files | Signal columns with timestamps | `FileReceiver` |
| Directory of files | Batch of any above | `FileDirReceiver` |
| Live stream | Continuous real-time feed | `StreamingReceiver` |

**Example `candump` line:**
```
(1609459200.123456) vcan0 0C6#00000000DEADBEEF
```

**Example CSV row (output.csv):**
```
timestamp,rpm,speed,throttle,...
1609459200,850,0,0,...
```

---

## 📊 Example Output

Detected anomalies are written to `anomalies.csv`:

```csv
timestamp,signal,value,predictor,severity
1609460012,rpm,9500,STATISTICAL,HIGH
1609460035,rpm,9800,LSTM,HIGH
```

The Python trainer produces an anomaly plot (`ano.png`) showing reconstruction error over time with a threshold line highlighting anomalous windows:

![Anomaly Detection Plot](./ano.png)

---

## ❓ FAQ

**Q: What CAN hardware is supported?**  
A: The project supports SocketCAN (Linux), Arduino-based CAN shields via serial proxy, and offline replay from `candump` log files. Any hardware that can produce `candump`-compatible output or send serial CAN frames will work.

**Q: Which car models are supported?**  
A: The repository includes a concrete `Mazda6CarData` implementation. Other vehicles can be supported by subclassing `ICarData` and mapping the relevant CAN IDs and signal decoding logic.

**Q: How does the SLTL predictor work?**  
A: SLTL (Signal Linear Temporal Logic) allows you to express temporal properties of vehicle signals as logical formulas — for example, *"speed must increase within N frames after RPM increases"*. A concrete example is implemented in `SpeedIncreasesAfterRPMIncreasesProperty`. Violations of these properties are flagged as anomalies.

**Q: Do I need a real car to test this?**  
A: No. The included `bad_rpm.csv` and `output.csv` datasets, plus `candump` replay mode, allow full offline testing without any vehicle hardware.

**Q: How is the LSTM model integrated into the C++ runtime?**  
A: The `AnomalyPredictorLSTM` class in C++ handles inference. The trained model (exported from `LSTMAnomaly.py` and stored in the `lstm/` directory) is loaded at runtime. The exact inference mechanism (TensorFlow C API, ONNX, or subprocess call) can be confirmed in `anomalypredictorlstm.cpp`.

**Q: Why are there `.vs`, `.vscode`, and `.qtc_clangd` folders committed?**  
A: These are IDE configuration folders for Visual Studio, VS Code, and Qt Creator respectively. They were not excluded in the `.gitignore`. It is safe to delete them if you use a different IDE.

**Q: Can I add a new anomaly detection algorithm?**  
A: Yes. Subclass `AnomalyPredictor` (see `anomalypredictor.h`), implement the required interface, and register your class in `main.cpp` or the processor thread setup.

---

## 👤 Author

**4lister**  
GitHub: [@4lister](https://github.com/4lister)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).
