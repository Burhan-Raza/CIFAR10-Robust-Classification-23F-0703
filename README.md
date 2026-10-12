# Robust CIFAR-10 Classification with 500 Training Images

Deep Learning for Perception, Assignment 2,(Fall 2026)
Roll No: 23F-0703

Train a small CNN with an Inception module from scratch on 500 CIFAR-10 images (400 train / 100 validation) and test it on a corrupted ("shifted") version of the CIFAR-10 test set.

## Repository contents

| File | Description |
|---|---|
| `DL_ASS02_23F_0703.ipynb` | Complete, executed notebook (Parts A to F, all outputs visible) |
| `requirements.txt` | Python packages needed |
| `README.md` | This file |

## Results (mean ± std over seeds 0, 1, 2)

| Configuration | Val acc % | Clean test % | Shifted test % |
|---|---|---|---|
| No augmentation | 51.7 ± 1.2 | 41.5 ± 0.4 | 18.0 ± 0.5 |
| Geometric only | 50.7 ± 1.2 | 41.2 ± 0.4 | 17.9 ± 0.8 |
| Photometric only | 47.3 ± 2.4 | 35.1 ± 3.0 | 18.1 ± 1.2 |
| Geometric + photometric | 57.7 ± 1.2 | 42.9 ± 1.6 | 22.3 ± 1.2 |
| + Cutout | 55.7 ± 2.1 | 42.2 ± 0.9 | 23.1 ± 1.0 |
| + CutMix + label smoothing | 52.7 ± 0.9 | 41.1 ± 0.8 | 21.2 ± 0.4 |
| **Final model (tuned)** | 56.0 ± 0.0 | 41.4 ± 0.7 | **22.5 ± 0.4** |

Model size: 64,402 trainable parameters (limit 1,000,000).

## How to reproduce

### 1. Setup

Python 3.10+ is recommended. A GPU is optional (each training run takes about 5 to 10 seconds on a GPU).

```bash
git clone <your-repo-link>
cd <repo-folder>
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook DL_ASS01_23F_0703.ipynb
```

In Jupyter choose **Kernel > Restart & Run All**. Cells must run from top to bottom.
On Google Colab, upload the notebook, select a GPU runtime and use **Runtime > Run all**.

### 3. What happens when you run it

- CIFAR-10 is downloaded automatically by `torchvision` (about 170 MB) and cached as `cifar10.npz`.
- 500 images are drawn (50 per class, split seed 42) and split into 400 train / 100 validation.
- The shifted test set is generated inside the notebook from the 10,000 clean test images (seed 1234): rotation up to ±25°, brightness/contrast change, blur, noise, a hidden colour-channel permutation, and about 25% rectangular occlusion.
- All training runs use seeds 0, 1 and 2, with at most 60 epochs.
- Results are cached in `results/` as `.pkl` files.

### 4. Training from scratch (no cache)

The notebook has `USE_CACHE = True`. To retrain every run instead of loading saved results, set `USE_CACHE = False` in the cell that defines it, or delete the `results/` folder. Small differences in the last decimal can appear across different GPUs and PyTorch versions.

## Rules followed

- No pretrained weights, trained from scratch.
- At most 1,000,000 trainable parameters, with an Inception-style module (1×1, 3×3, 5×5 branches with 1×1 reductions).
- Hyper-parameters chosen on the 100-image validation split only. Test sets are used only for reporting.
- Augmentation is applied on the fly to training data only. No augmented copies are saved.
- Fixed random seeds, three runs, mean ± std reported.

## Notebook outline

1. Setup, seeds, data loading and shifted test construction
2. Model (Inception network)
3. Augmentation functions
4. Part A: baseline and learning curves
5. Part B: augmentation experiments (vertical flip, rotation ±90°, too much augmentation)
6. Part C: validation-only hyper-parameter search
7. Part D: ablation table
8. Part E: per-class accuracy, confusion matrix, misclassified images, hidden-corruption test
9. Part F: test-time augmentation (bonus)
