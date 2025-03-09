from setuptools import setup, find_packages

setup(
    name="stateful_simulation_gaussian",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.3.0",
        "wandb>=0.12.0",
        "tqdm>=4.60.0",
        "plotly>=5.0.0",
        "pillow>=8.0.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
)