# Band Gap Prediction via Machine Learning on Materials Project Data

**Course:** Machine Learning for Materials Science | Connecticut College  
**Author:** Dylan Field | May 2026
**Format:** Jupyter Notebook File (Using Python)

## Overview

This project builds and compares two machine learning models for predicting the electronic band gap (in eV) of inorganic crystalline materials. Band gap is a critical property for identifying semiconductor candidates in solar cells and LEDs, but computing it via Density Functional Theory (DFT) takes hours to days per material. This model enables fast screening at scale.

## Dataset

- **Source:** [Materials Project](https://materialsproject.org) API (~50,000 entries pulled)
- **After filtering:** 18,608 semiconductors and insulators (band gap > 0 eV)
- **Features:** 132 Magpie compositional features via `matminer`, expanded to 8,778 dimensions via degree-2 polynomial interaction terms
- **Target:** `band_gap` (continuous, range: ~0 to 17.9 eV, mean: 1.96 eV)

## Methods

### Exploratory Data Analysis

- PCA: 60 components needed to explain 90% of variance across 8,778 features
- UMAP + KMeans (k=4): 4 loosely separated material clusters (likely oxides, sulfides, halides, and nitrides), with significant overlap due to the absence of crystal structure information

### Models

| Model | Architecture | Feature Input |
|---|---|---|
| Random Forest | 100 trees, bootstrap sampling | 132 Magpie features (unscaled) |
| Residual MLP | 2 ResNet-style skip-connection blocks, Adam optimizer, 50 epochs | 8,778 polynomial features (standardized) |

### Results

| Model | MAE (eV) | RMSE (eV) |
|---|---|---|
| Random Forest | **0.5596** | **0.8231** |
| Residual MLP | 0.6503 | 0.9427 |

Random Forest outperformed the Residual MLP, primarily due to overfitting in the MLP on noisy polynomial features and the well-documented strength of tree ensembles on tabular data.

## Setup

```bash
pip install numpy pandas matplotlib mp-api matminer pymatgen scikit-learn umap-learn tensorflow keras
```

A [Materials Project API key](https://next-gen.materialsproject.org/api) is required to re-pull the dataset.

## References

- He et al. (2016) ResNet skip connection architecture
- Keras Structured Data Examples: https://keras.io/examples/structured_data/
- Materials Project: https://materialsproject.org
