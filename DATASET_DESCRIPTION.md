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
# Synthetic Hemochromatosis Dataset

## Overview

This repository contains a **synthetic dataset** of 1 000 subjects (500 hemochromatosis cases, 500 controls) designed for educational and methodological research in biomedical data science. Every record was generated programmatically — no real patient data is included.

The dataset reproduces the clinically expected biomarker patterns of hereditary hemochromatosis (iron-overload disease) with realistic inter-group overlap, missing data, and correlated laboratory values.

## Variables (44 columns)

| Domain | Variables | Notes |
|---|---|---|
| **Demographics** | `age`, `sex` | Age 18-85; sex as biological binary |
| **Genetics** | `HFE_variant_present/id`, `HJV_variant_present/id`, `HAMP_variant_present/id`, `TFR2_variant_present/id`, `SLC40A1_variant_present/id` | Binary carrier flag + specific rsID (e.g., rs1800562 for C282Y) |
| **Complete blood count** | `hemoglobin`, `hematocrit`, `rbc_count`, `mcv`, `mch`, `mchc`, `rdw`, `wbc_count`, `neutrophils_pct/abs`, `lymphocytes_pct/abs`, `monocytes_pct/abs`, `eosinophils_pct/abs`, `basophils_pct/abs`, `platelet_count` | 18 CBC parameters with physiological correlations |
| **Iron panel** | `serum_iron`, `ferritin`, `tibc`, `uibc`, `transferrin_saturation` | Mathematically consistent (UIBC = TIBC - serum iron; TSAT = serum iron / TIBC x 100) |
| **Liver enzymes** | `ast`, `alt`, `ggt` | Elevated only in the subset of cases with hepatic involvement |
| **Fibrosis indices** | `apri`, `fib4`, `gpr` | Derived from primary labs using standard clinical formulas |
| **Outcome** | `diagnosis` | 1 = hemochromatosis, 0 = control |

Missing data is injected at ~14 % overall (MCAR), heavier in labs than in demographics, mimicking real clinical datasets.

## Predictive Model Comparison

Six models were evaluated using **5-fold stratified cross-validation** with median imputation and standard scaling where appropriate.

### Results

| Model | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| **Logistic Regression (top 10 features)** | **0.971** | **0.910** | 0.890 | **0.930** | **0.908** |
| Logistic Regression (all 37 features) | 0.969 | 0.904 | 0.886 | 0.922 | 0.902 |
| Random Forest | 0.970 | 0.909 | 0.902 | 0.916 | 0.908 |
| XGBoost | 0.970 | 0.905 | 0.900 | 0.910 | 0.904 |
| SVM (RBF kernel) | 0.962 | 0.904 | 0.882 | 0.926 | 0.902 |
| Neural Network (MLP) | 0.954 | 0.877 | 0.860 | 0.894 | 0.875 |

### Top 10 ranked features (used in the reduced logistic regression)

| Rank | Feature | Why it matters |
|---|---|---|
| 1 | `ferritin` | Primary biochemical marker of iron overload |
| 2 | `HFE_variant_present` | Causal genotype in >70 % of cases |
| 3 | `ast` | Reflects hepatic iron deposition |
| 4 | `serum_iron` | Directly elevated in iron overload |
| 5 | `uibc` | Inverse marker — low UIBC = saturated transferrin |
| 6 | `alt` | Co-elevated with AST in liver involvement |
| 7 | `tibc` | Decreased in iron overload |
| 8 | `transferrin_saturation` | Derived ratio; screening threshold typically >45 % |
| 9 | `ggt` | Additional hepatic injury marker |
| 10 | `sex` | Males present earlier and more severely |

## Why ML Models Do Not Outperform Logistic Regression

The best ML model (Random Forest, AUC = 0.970) shows virtually **no advantage** over logistic regression (AUC = 0.971). This is not a limitation of the models — it reflects the **pathophysiology of the disease itself**. Three factors explain the result:

### 1. The discriminating signal is concentrated in a few linearly separable biomarkers

Hemochromatosis diagnosis in clinical practice rests on a short, well-defined decision pathway:

> Elevated ferritin + elevated transferrin saturation + HFE genotype = diagnosis.

These three variables alone carry the majority of the predictive information. Their relationship to the outcome is **monotonic and roughly linear on the log/logit scale**: higher ferritin and higher TSAT increase the probability of disease in a smooth, graded fashion. There are no threshold effects, interactions, or non-linear "knees" that a tree or neural network could exploit beyond what a logistic function already captures.

### 2. No complex feature interactions to discover

In diseases where ML excels over logistic regression (e.g., image-based diagnosis, multi-organ syndromes, polygenic risk with epistasis), the discriminative pattern involves **high-order interactions** between many variables. In hemochromatosis:

- The CBC parameters (hemoglobin, MCV, WBC, etc.) are largely **non-discriminating** — iron overload does not markedly alter blood counts until very late stages.
- The liver enzymes (AST, ALT, GGT) are elevated only in a **subset** of cases with hepatic involvement, adding modest incremental value.
- The fibrosis indices (APRI, FIB-4, GPR) are **deterministic functions** of the primary labs, so they add no independent information.
- The rarer genetic variants (HJV, HAMP, TFR2, SLC40A1) have very low prevalence in both groups and contribute minimally.

Without meaningful interactions to learn, ensemble and deep learning methods simply converge to solutions that are **functionally equivalent** to logistic regression.

### 3. Sample size favors simpler models

With n = 1 000 and 37 features, high-capacity models (MLP, deep ensembles) risk **overfitting** to noise. Indeed, the MLP (AUC = 0.954) performs worst among all models. Logistic regression's inductive bias — assuming a linear decision boundary — is well-matched to the true data-generating process, giving it an inherent advantage in this regime.

### Clinical implication

This result mirrors findings in the real hemochromatosis literature: simple laboratory criteria (ferritin > 300 ng/mL in men or > 200 ng/mL in women, TSAT > 45 %, confirmed by HFE genotyping) achieve diagnostic sensitivity and specificity above 90 %. Sophisticated ML pipelines offer **no clinically meaningful improvement** for this particular disease, which is precisely why international guidelines still recommend straightforward algorithmic screening rather than predictive models.

For diseases where the decision boundary is inherently nonlinear — radiology, multi-omics integration, polygenic risk scoring — ML methods would be expected to show a clear advantage.

## Reproducibility

```bash
# Generate the dataset
python generate_dataset.py

# Run the full model comparison
python model_evaluation.py
```

Both scripts use `numpy.random.seed(42)` for full reproducibility.
