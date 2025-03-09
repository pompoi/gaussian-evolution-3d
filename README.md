# Gaussian Evolution 3D

Dynamic 3D scene reconstruction using Gaussian splatting and Liquid Neural Networks for temporal evolution.

## Overview

This project implements a novel approach to reconstructing and rendering time-evolving 3D scenes using Gaussian splatting as the core representation. The model combines:
- **3D Gaussian Primitives** for scene representation
- **Liquid Neural Networks** for temporal evolution
- **Differentiable Rendering** for end-to-end training

## Features

- **Dynamic Scene Representation**: Model time-evolving 3D scenes using adaptive Gaussian primitives
- **Temporal Evolution**: Use Liquid Neural Networks to predict scene changes over time
- **Multi-View Synthesis**: Render novel viewpoints for any trained time step
- **Efficient Training**: 
  - Truncated Backpropagation Through Time (TBPTT)
  - Multi-GPU support with DDP
  - Memory-efficient implementation
- **Comprehensive Visualization**: Training progress monitoring with Weights & Biases integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pompoi/gaussian-evolution-3d.git
cd gaussian-evolution-3d
```

2. Install the package:
```bash
pip install -e .
```

## Data Structure

The training dataset should be organized as follows:
```
data_dir/
├── time_000/
│   ├── rgb_00.png
│   ├── depth_00.png
│   ├── mask_00.png (optional)
│   ├── info.json
│   └── ...
├── time_001/
└── ...
```

Each time step folder contains:
- RGB images from multiple views
- Depth maps
- Optional alpha masks
- Camera parameters in `info.json`

## Training

### Single GPU Training

```bash
python scripts/train_single_gpu.py \
    --config configs/default_config.yaml \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --num_epochs 100 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --use_wandb
```

### Multi-GPU Training

```bash
python scripts/train_multi_gpu.py \
    --config configs/default_config.yaml \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --num_epochs 100 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --use_wandb \
    --num_gpus 4  # or -1 to use all available GPUs
```

## Configuration

The model can be configured through a YAML file. See `configs/default_config.yaml` for available options:

```yaml
model:
  num_gaussians: 10000
  lnn_hidden_size: 256
  lnn_num_layers: 3
  ...

renderer:
  image_width: 800
  image_height: 600
  ...

training:
  sequence_length: 60
  sequence_stride: 30
  ...

loss:
  l1_weight: 1.0
  ssim_weight: 0.5
  ...
```

## Project Structure

```
gaussian-evolution-3d/
├── src/
│   ├── models/          # Neural network models
│   ├── renderer/        # Gaussian splatting renderer
│   ├── training/        # Training utilities
│   ├── data/           # Dataset handling
│   └── utils/          # Helper functions
├── scripts/            # Training scripts
├── configs/           # Configuration files
└── tests/            # Unit tests
```

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@misc{gaussian-evolution-3d,
  author = {Your Name},
  title = {Gaussian Evolution 3D},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/pompoi/gaussian-evolution-3d}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.