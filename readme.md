# RoboMerge: Universal Robot Data Standardization Pipeline

RoboMerge is a preprocessing pipeline that standardizes diverse robot data sources into a common format ready for FAST tokenization. It solves the upstream data preparation challenge for robot learning systems by providing a unified interface for data ingestion, standardization, and quality validation.

## Features

- Data format standardization across robot platforms
- Frequency normalization (15Hz → 50Hz)
- Action space mapping and normalization
- Automated quality validation
- FAST-ready data output
- Extensible configuration system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ShreshthRajan/robomerge.git
cd robomerge
```

2. Set up environment:

Using conda (recommended):
```bash
# For Mac with M1/M2
conda create -n tf_env python=3.10
conda activate tf_env
pip install -e .

# Note: The setup will automatically handle Mac-specific TensorFlow requirements
```

3. Download DROID dataset:
```bash
# Install gsutil if needed
pip install gsutil

# Create data directory
mkdir -p data/droid/1.0.0

# Download DROID dataset
gsutil -m cp -r "gs://gresearch/robotics/droid_100/*" ./data/droid/1.0.0/
```

## Environment Requirements

- Python 3.10.16 or later
- TensorFlow 2.16.2
- NumPy 1.26.4
- Matplotlib 3.10.0
- h5py 3.12.1

Note: For Mac M1/M2 users, the setup includes tensorflow-metal and tensorflow-macos packages automatically.

[Rest of README remains the same...]
