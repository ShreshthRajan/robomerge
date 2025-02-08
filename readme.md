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

# Download DROID dataset
gsutil -m cp -r "gs://gresearch/robotics/droid_100/*" data/droid/1.0.0/
```

## Demo Notebook
The demo.ipynb notebook provides an end-to-end demonstration of RoboMerge's capabilities:

1. Data ingestion from DROID format
2. Standardization and frequency normalization
3. Quality validation with metrics
4. FAST preprocessing
5. Extensibility demonstration

To run the demo:
1. Ensure you've completed the installation steps above
2. Launch Jupyter:
```bash
        bashCopyjupyter notebook
```
3. Open demo.ipynb
4. Run all cells

## Extending to New Robots
RoboMerge uses a configuration-based system for adding support for new robots. 

Example configuration:
```bash
pythonCopyrobot_config = {
    "name": "new_robot",
    "input_format": {
        "type": "csv",
        "delimiter": ",",
        "columns": ["timestamp", "joint1", "joint2", "joint3"]
    },
    "action_space": {
        "dimensions": 3,
        "type": "joint_velocity",
        "limits": [-1.0, 1.0]
    },
    "control_frequency": 100
}
```

## Environment Requirements
- Python 3.10.16 or later
- TensorFlow 2.16.2
- NumPy 1.26.4
- Matplotlib 3.10.0
- h5py 3.12.1
- gsutil (for data download)