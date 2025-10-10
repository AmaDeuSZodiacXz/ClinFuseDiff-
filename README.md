# ClinFuseDiff

**Diffusion-Based Test-Time Training for Robust Multimodal Clinical Data Fusion**

## Overview

ClinFuseDiff combines diffusion models with test-time training (TTT) to enable robust multimodal clinical data fusion that adapts to missing modalities at test time.

## Key Features

- **Diffusion-based fusion**: Uses conditional diffusion models to generate robust fused representations
- **Test-time adaptation**: Adapts to missing modalities through self-supervised learning
- **Modality imputation**: Generates plausible features for missing modalities
- **Clinical applications**: Designed for real-world clinical scenarios with incomplete data

## Project Structure

```
ClinFuseDiff/
├── src/
│   ├── models/          # Model implementations
│   ├── data/            # Data loaders and preprocessing
│   ├── utils/           # Utility functions
│   └── training/        # Training and evaluation scripts
├── experiments/         # Experimental results
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks for analysis
└── docs/               # Documentation
```

## Installation

```bash
# Create conda environment
conda create -n clinfusediff python=3.9
conda activate clinfusediff

# Install dependencies
pip install -r requirements.txt
```

## Usage

Coming soon...

## Citation

If you use this code, please cite:

```bibtex
@article{clinfusediff2026,
  title={ClinFuseDiff: Diffusion-Based Test-Time Training for Robust Multimodal Clinical Data Fusion},
  author={TBD},
  journal={ICLR},
  year={2026}
}
```

## License

MIT License
