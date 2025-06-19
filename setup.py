from setuptools import setup, find_packages
import platform

# Define base requirements
requirements = [
    "numpy>=1.26.4",
    "matplotlib>=3.10.0", 
    "h5py>=3.12.1",
    "tensorflow-io>=0.37.1"
]

# Add platform-specific requirements
if platform.system() == "Darwin" and platform.processor() == "arm":
    # For Mac M1/M2
    requirements.extend([
        "tensorflow-macos>=2.16.2",
        "tensorflow-metal>=1.2.0"
    ])
else:
    # For other platforms
    requirements.append("tensorflow>=2.16.2")

setup(
    name="robomerge",
    version="0.3.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10.16",
    author="Shreshth Rajan",
    description="A preprocessing pipeline for standardizing robot data for FAST and Knowledge Insulation (KI) training",
    long_description="""
    RoboMerge: Enhanced Robot Data Preprocessing Pipeline
    
    Features:
    - FAST tokenization for efficient robot action representation
    - Knowledge Insulation (KI) support for π₀.₅ + KI training
    - Dual training objectives: discrete tokens + continuous actions
    - Web data mixing for improved generalization
    - Gradient isolation for preserving VLM knowledge
    - Real-time compatible preprocessing
    
    Compatible with Physical Intelligence's latest VLA architectures.
    """,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    keywords="robotics, machine learning, VLA, FAST, knowledge insulation, physical intelligence",
    project_urls={
        "Documentation": "https://github.com/shreshth-rajan/robomerge",
        "Source": "https://github.com/shreshth-rajan/robomerge",
    }
)