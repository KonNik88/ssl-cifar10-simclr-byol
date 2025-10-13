# Self-Supervised Learning on CIFAR-10 (SimCLR & BYOL)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Minimal, portfolio-ready project to demonstrate **Self-Supervised Learning (SSL)** using **SimCLR** and **BYOL** on the CIFAR-10 dataset.  
The project provides a clean and reproducible PyTorch pipeline with evaluation protocols and visualizations.

---

## What's inside
- **Implementations** of SimCLR and BYOL in PyTorch (with PyTorch Lightning).
- **Evaluation protocols:**
  - Linear probe (frozen encoder + logistic regression / linear layer).
  - k-NN monitor during training.
  - Semi-supervised evaluation with limited labels.
- **Visualizations:**
  - UMAP/t-SNE plots of learned embeddings.
  - Nearest neighbors retrieval with FAISS.
  - Augmentation examples for CIFAR-10.

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/KonNik88/ssl-cifar10-simclr-byol.git
cd ssl-cifar10-simclr-byol
```

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # (Linux/Mac)
.venv\Scripts\activate           # (Windows)

pip install -r requirements.txt
```

### 3. Train a model
Run SimCLR with ResNet-18 backbone:
```bash
python -m src.train_ssl --config configs/simclr_resnet18.yaml
```

Run BYOL with ResNet-18 backbone:
```bash
python -m src.train_ssl --config configs/byol_resnet18.yaml
```

### 4. Evaluate
- Linear probe:
```bash
python -m src.eval_linear --ckpt artifacts/checkpoints/simclr_r18.pt
```

- k-NN:
```bash
python -m src.eval_knn --ckpt artifacts/checkpoints/byol_r18.pt
```

---

## Results (expected)
- SSL improves representation quality compared to random initialization.
- Linear probe achieves significantly higher accuracy than supervised baselines with limited labels.
- Embedding space clusters semantically similar images (see UMAP plots).

*(Detailed results, tables, and figures will be added after experiments.)*

---

## Project Structure
```
.
├─ configs/           # YAML configs for experiments
├─ src/               # Source code (data, models, training, evaluation)
├─ notebooks/         # Visualization and analysis
├─ artifacts/         # Saved checkpoints, logs, metrics
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ .gitignore
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements
- [SimCLR paper (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)  
- [BYOL paper (Grill et al., 2020)](https://arxiv.org/abs/2006.07733)  
- [PyTorch Lightning](https://lightning.ai/) for clean training loops.
