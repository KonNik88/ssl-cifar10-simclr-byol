# Self-Supervised Learning on STL-10 (SimCLR) — Portfolio Project

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/lightning-2.x-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean, reproducible **Self-Supervised Learning (SSL)** project that demonstrates **SimCLR** pretraining on **STL-10 unlabeled (100k images)** and evaluates learned representations with standard protocols:

- **kNN@K** on frozen embeddings  
- **Linear probe** (frozen encoder + linear classifier)  
- **UMAP** visualization of embedding space  
- **Nearest-neighbor retrieval** in embedding space (cosine similarity)

This repo is designed to be **portfolio-ready**:
- runs on a single GPU (e.g., RTX 2070),
- produces structured artifacts (logs / checkpoints / metrics),
- keeps training in scripts and analysis in notebooks.

---

## TL;DR (final results)

**Strong SimCLR run:** `simclr_version_4` (50 epochs)

- **kNN@20 accuracy:** **0.7405**
- **Linear-probe accuracy (20 epochs):** **0.7360**

All results are reproducible from artifacts in `artifacts/` and summarized in:
- `artifacts/metrics/runs_index.csv`
- `artifacts/metrics/summary.csv`

---

## Project overview

### Training (scripts)
- Train SimCLR on **STL-10 unlabeled** using `src/train_ssl.py`
- Logs go to `artifacts/logs/…`
- Checkpoints go to `artifacts/checkpoints/…`
- Run registry + aggregated metrics go to `artifacts/metrics/…`

### Evaluation (scripts)
- `src/eval_knn.py` — kNN on frozen embeddings (STL-10 train → test)
- `src/eval_linear.py` — linear probe on frozen embeddings

### Analysis (notebooks)
- `01_augmentations_preview.ipynb` — why augmentations matter in SSL
- `02_experiments_report_fixed.ipynb` — training curves / loss analysis
- `03_umap_embeddings_fixed.ipynb` — compute embeddings + UMAP visualization
- `04_retrieval_demo.ipynb` — nearest-neighbor retrieval + Hit@10 sanity check
- `05_ssl_final_simclr.ipynb` — **final showcase** (all key results in one notebook)

---

## Quickstart

### 1) Create environment

**Option A: conda (recommended)**
```bash
conda env create -f environment.yml
conda activate ssl_env
```

**Option B: pip**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run: training → evaluation → final notebook

### 2) Train SimCLR (strong config)

```bash
python -m src.train_ssl --config configs/simclr_r18_stl10_strong.yaml
```

This will create:
- `artifacts/logs/simclr/version_X/metrics.csv`
- `artifacts/checkpoints/simclr/simclr_version_X/{last.ckpt,best.ckpt}`
- update `artifacts/metrics/runs_index.csv` and `artifacts/metrics/summary.csv`

### 3) Evaluate representation quality

**kNN@20**
```bash
python -m src.eval_knn --project-root . --k 20 --use best
```

**Linear probe (20 epochs)**
```bash
python -m src.eval_linear --project-root . --epochs 20 --use best
```

After running these scripts, open:
- `artifacts/metrics/summary.csv` (updated with `knn_acc` and `linear_acc`)

### 4) Open final showcase notebook

Run Jupyter and open:

- `notebooks/05_ssl_final_simclr.ipynb`

This notebook reproduces:
- training curves (loss),
- kNN + linear-probe metrics,
- UMAP embeddings,
- retrieval demo + Hit@10 sanity metric.

---

## Reproducibility notes

- Paths are handled relative to the project root (`PROJECT_ROOT` in notebooks).
- The repo stores:
  - **checkpoints** (`best.ckpt`, `last.ckpt`)
  - **metrics logs** (`metrics.csv`)
  - **run registry** (`runs_index.csv`)
  - **aggregated summary** (`summary.csv`)
- Notebooks are analysis-only: they **do not train** models.

---

## Project structure

```
D:\ML\SSL
├── artifacts/
│   ├── checkpoints/
│   ├── embeddings/
│   ├── figures/
│   ├── logs/
│   └── metrics/
├── configs/
├── data/
├── lightning_logs/          # optional (legacy Lightning dir if used)
├── notebooks/
└── src/
    ├── data/
    ├── losses/
    ├── models/
    └── utils/
```

---

## Future work (optional)

- Add **BYOL** (non-contrastive SSL) and compare side-by-side with the same metrics (kNN + linear probe + UMAP + retrieval).
- Add FAISS indexing for large-scale retrieval (engineering upgrade; not required for this portfolio version).

---

## References

- **SimCLR**: Chen et al., 2020 — *A Simple Framework for Contrastive Learning of Visual Representations*  
- **STL-10** dataset: Coates et al., 2011  
- **PyTorch Lightning** for clean training loops

---

## License

MIT License. See [LICENSE](LICENSE) for details.
