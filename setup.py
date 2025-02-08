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
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10.16",
    author="Shreshth Rajan",
    description="A preprocessing pipeline for standardizing robot data for FAST",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)