# ZebraLSTM

ZebraLSTM is an advanced Long Short-Term Memory (LSTM) architecture designed to optimize the performance of sequential data processing tasks. By dividing the hidden state into multiple parallel stripes, ZebraLSTM enhances computational efficiency and scalability, particularly when leveraging modern GPU architectures.

## Features
- Striped hidden state processing for parallelism and efficiency
- Configurable architecture for various applications
- Suitable for NLP, time-series analysis, and drug design

## Directory Structure
- `src/`: Source code for ZebraLSTM
- `examples/`: Example scripts and notebooks
- `tests/`: Unit tests and integration tests
- `docs/`: Documentation and references
- `cuda/`: CUDA implementations and related scripts
- `scripts/`: Utility scripts for setup and installation

## Installation
To install dependencies, run:
```bash
pip install -r requirements.txt
