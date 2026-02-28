# Synthetic Schizophrenia Biomarker Dataset

## Overview

A synthetic dataset of **1,000 individuals** (500 schizophrenia cases, 500 healthy controls) designed to evaluate machine learning vs. traditional statistical models for biomarker-based psychiatric classification.

The dataset contains **39 columns** spanning demographics, genetic variants, amino acids, lipids, redox markers, and inflammatory cytokines — all with physiologically plausible ranges drawn from published schizophrenia biomarker literature.

## Features

| Category | Variables | Examples |
|----------|-----------|---------|
| Demographics | 2 | Age (18–70), sex |
| Genetic variants | 16 (8 genes × 2) | C4A, DRD2, GRM3, GRIN2A, SLC39A8, DISC1, COMT, BDNF — binary presence + rsID |
| Amino acids & metabolites | 8 | Serum glutamate, serine, alanine, lactate, citrate, ornithine, citrulline, arginine |
| Lipid panel | 6 | Triglycerides, LDL, HDL, VLDL, phospholipids, sphingolipids |
| Redox markers | 2 | Reduced glutathione (GSH), nitric oxide (NO) |
| Inflammatory cytokines | 3 | IL-6, TNF-α, IL-8 |

Missing data: ~5% MCAR across numeric biomarker columns.

## Expected Model Performance (5-Fold Stratified CV)

| Model | AUC | Accuracy |
|-------|-----|----------|
| **SVM (RBF)** | **0.761** | 0.681 |
| Neural Network (MLP) | 0.752 | 0.673 |
| XGBoost | 0.744 | 0.674 |
| Random Forest | 0.738 | 0.670 |
| Logistic Regression (all features) | 0.675 | 0.610 |
| Logistic Regression (top 10) | 0.665 | 0.615 |

**ML advantage over logistic regression: +8.5% AUC.**

These AUCs (0.67–0.76) are consistent with published schizophrenia biomarker studies, which typically report AUCs in the 0.70–0.85 range.

## How the ML Advantage Was Engineered

### The problem with naïve synthetic data

If you generate cases and controls with shifted biomarker means (e.g., cases have higher glutamate), logistic regression captures those marginal differences just as well as any ML model. The result is that all models perform similarly, which is unrealistic — real biomedical data often contains nonlinear structure that ML exploits.

### Solution: XOR-like gene × biomarker interactions

The dataset uses a two-layer signal architecture:

#### Layer 1 — Weak linear main effects (what logistic regression sees)

Each biomarker has a tiny case–control mean shift (Cohen's d ≈ 0.08–0.15). Individually, no single feature is strongly predictive. Cumulatively across 29 features, logistic regression achieves AUC ≈ 0.67.

#### Layer 2 — Strong nonlinear interactions (what ML models discover)

The **direction** of biomarker shifts **reverses** depending on genetic context. This creates signal in the joint distribution that is invisible in the marginals:

| Gene | Gene+ Cases | Gene− Cases | Net Marginal |
|------|-------------|-------------|--------------|
| **COMT** | Glutamate ↑↑, GSH ↓↓ | Glutamate ↓, GSH ↑ | ≈ 0 (cancels) |
| **BDNF** | IL-6 ↑↑, TNF-α ↑↑, IL-8 ↑↑ | IL-6 ↓, TNF-α ↓ | ≈ 0 (cancels) |
| **DRD2** | Triglycerides ↑↑, HDL ↓↓ | Triglycerides ↓, HDL ↑ | ≈ 0 (cancels) |

**Why logistic regression fails**: LR assigns a single coefficient to each feature. If glutamate is high for COMT+ cases but low for COMT− cases, the average effect across all cases is near zero — LR's coefficient for glutamate is close to zero.

**Why ML models succeed**: A decision tree can learn split rules like:
- "If COMT = 1 AND glutamate > 58 → case"
- "If COMT = 0 AND glutamate < 42 → case"

SVMs with RBF kernels and neural networks can similarly learn these nonlinear decision boundaries.

### Additional nonlinear signal sources

- **Epistatic amplifiers**: Two-gene combinations (C4A × COMT, DRD2 × BDNF, GRIN2A × GRM3) stack additional shifts on top of the single-gene XOR axes.
- **Age × gene interaction**: Young (<30) gene-carrier cases show amplified inflammatory and oxidative stress signatures.
- **Controls receive ghost noise**: Tiny, symmetric random noise is added to controls with the same genotypes to prevent trivial gene-only classification shortcuts.

### Biological plausibility

The interaction structure is motivated by real biology:
- **COMT** (catechol-O-methyltransferase) regulates prefrontal dopamine clearance; its Val158Met polymorphism is one of the most studied variants in schizophrenia.
- **BDNF** (brain-derived neurotrophic factor) modulates neuroinflammation and synaptic plasticity.
- **DRD2** (dopamine receptor D2) is the primary target of antipsychotic medications and is linked to metabolic side effects.
- Gene × diagnosis conditional penetrance (where a variant only manifests phenotypically in the disease state) is a well-established concept in psychiatric genetics.

## Files

| File | Description |
|------|-------------|
| `generate_schizophrenia_dataset.py` | Dataset generator (run to regenerate) |
| `evaluate_schizophrenia_models.py` | Model training and evaluation pipeline |
| `schizophrenia_synthetic_dataset.csv` | The generated dataset (1,000 × 39) |
| `schizophrenia_data_dictionary.csv` | Variable definitions, types, and units |

## Reproducibility

All random number generation uses `SEED = 42`. Running `python generate_schizophrenia_dataset.py` will produce an identical dataset. Model evaluation uses the same seed for cross-validation splits.
