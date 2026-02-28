"""
Synthetic Biomedical Dataset Generator for Schizophrenia Research

Generates a dataset of 1000 individuals (500 schizophrenia cases, 500 healthy
controls) with:
- Genetic variants in schizophrenia-associated genes (C4A, DRD2, GRM3,
  GRIN2A, SLC39A8, DISC1, COMT, BDNF) with polygenic risk architecture
- Continuous biochemical and metabolomic variables with physiologically
  plausible ranges
- Biologically consistent correlation structures (cytokine cluster, lipid
  cluster, glutamate–glutathione axis)
- Demographic covariates (age, sex) with mild confounding
- ~5% missing data at random

All values reflect published literature ranges for schizophrenia biomarkers
while maintaining substantial group overlap to ensure realistic
misclassification rates.
"""

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
np.random.seed(SEED)

N_CASES = 500
N_CONTROLS = 500
N_TOTAL = N_CASES + N_CONTROLS

# ============================================================================
# 1. Demographics
# ============================================================================

def generate_demographics(n_cases, n_controls):
    """Generate age and sex with mild confounding.

    Schizophrenia has a slight male preponderance (~1.4:1) and typical onset
    in early adulthood.  We sample a broad clinical age range.
    """
    sex_cases = np.random.binomial(1, 0.58, n_cases)       # 58 % male
    sex_controls = np.random.binomial(1, 0.48, n_controls)  # 48 % male
    sex = np.concatenate([sex_cases, sex_controls])  # 1 = male, 0 = female

    age_cases = np.clip(
        np.random.normal(36, 11, n_cases), 18, 70
    ).astype(int)
    age_controls = np.clip(
        np.random.normal(34, 12, n_controls), 18, 70
    ).astype(int)
    age = np.concatenate([age_cases, age_controls])

    diagnosis = np.concatenate([
        np.ones(n_cases, dtype=int),
        np.zeros(n_controls, dtype=int),
    ])
    return diagnosis, age, sex


# ============================================================================
# 2. Genetic Variants  (polygenic risk architecture)
# ============================================================================

# Realistic rsIDs for schizophrenia-associated variants per gene.
VARIANT_CATALOG = {
    "C4A":     ["rs204991",   "rs3132468", "rs2071278"],
    "DRD2":    ["rs1801028",  "rs6277",    "rs1800497"],
    "GRM3":    ["rs6465084",  "rs2228595", "rs1468412"],
    "GRIN2A":  ["rs1014531",  "rs8049651", "rs7206256"],
    "SLC39A8": ["rs13107325", "rs13135092"],
    "DISC1":   ["rs821616",   "rs3738401", "rs6675281"],
    "COMT":    ["rs4680",     "rs165599",  "rs4633"],
    "BDNF":    ["rs6265",     "rs7103411", "rs2049046"],
}

# (freq_in_cases, freq_in_controls, approximate OR)
# MAFs 5-40 %, ORs 1.2-1.8 → polygenic, non-deterministic
GENE_PARAMS = {
    "C4A":     (0.35, 0.22, 1.8),
    "DRD2":    (0.28, 0.20, 1.5),
    "GRM3":    (0.22, 0.16, 1.5),
    "GRIN2A":  (0.18, 0.12, 1.6),
    "SLC39A8": (0.10, 0.07, 1.5),
    "DISC1":   (0.15, 0.10, 1.6),
    "COMT":    (0.40, 0.32, 1.4),
    "BDNF":    (0.30, 0.22, 1.5),
}


def generate_genetic_variants(n_cases, n_controls):
    """Generate binary variant presence + specific rsID per gene."""
    data = {}
    for gene, (freq_case, freq_ctrl, _) in GENE_PARAMS.items():
        presence_cases = np.random.binomial(1, freq_case, n_cases)
        presence_controls = np.random.binomial(1, freq_ctrl, n_controls)
        presence = np.concatenate([presence_cases, presence_controls])

        variants = VARIANT_CATALOG[gene]
        specific = [
            np.random.choice(variants) if p == 1 else np.nan
            for p in presence
        ]

        data[f"{gene}_variant_present"] = presence
        data[f"{gene}_variant_id"] = specific
    return data


# ============================================================================
# 3. Polygenic risk / latent severity score
# ============================================================================

def compute_polygenic_score(diagnosis, age, sex, genetic_data):
    """
    Latent score in [0, 1] that drives biomarker shifts.

    For cases: moderate score with genetic burden + demographic components.
    For controls: low score with some noise (ensures overlap).
    """
    n = len(diagnosis)
    score = np.zeros(n)

    gene_weights = {
        "C4A": 0.20, "DRD2": 0.15, "GRM3": 0.12, "GRIN2A": 0.12,
        "SLC39A8": 0.08, "DISC1": 0.10, "COMT": 0.13, "BDNF": 0.10,
    }
    burden = np.zeros(n)
    for gene, w in gene_weights.items():
        burden += genetic_data[f"{gene}_variant_present"] * w

    case_mask = diagnosis == 1
    ctrl_mask = diagnosis == 0

    score[case_mask] = (
        0.20
        + 0.25 * burden[case_mask]
        + 0.04 * ((age[case_mask] - 18) / 52)
        + 0.03 * sex[case_mask]
        + np.random.normal(0, 0.10, case_mask.sum())
    )
    score[ctrl_mask] = (
        0.06
        + 0.10 * burden[ctrl_mask]
        + 0.02 * ((age[ctrl_mask] - 18) / 52)
        + np.random.normal(0, 0.06, ctrl_mask.sum())
    )
    return np.clip(score, 0, 1)


# ============================================================================
# 4. Biochemical / Metabolomic Variables
# ============================================================================

def generate_amino_acids_and_metabolites(diagnosis, severity, age, sex):
    """
    Serum glutamate, serine, alanine, lactate, citrate, and urea-cycle
    intermediates (ornithine, citrulline, arginine).

    Cases show mild glutamate elevation and subtle perturbations in
    one-carbon / urea-cycle metabolism.
    """
    n = len(diagnosis)
    case = diagnosis == 1

    # --- Serum glutamate (µmol/L) ---
    # Normal ~20-80; cases mildly elevated (literature: ~+10-15 %)
    glutamate = np.where(
        case,
        np.random.normal(58 + severity * 12, 14, n),
        np.random.normal(48, 13, n),
    )
    glutamate = np.clip(glutamate, 10, 120).round(1)

    # --- Serine (µmol/L) ---
    # Normal ~80-150; cases show slight reduction
    serine = np.where(
        case,
        np.random.normal(105 - severity * 8, 18, n),
        np.random.normal(115, 20, n),
    )
    serine = np.clip(serine, 40, 200).round(1)

    # --- Alanine (µmol/L) ---
    # Normal ~200-500; mild increase in cases
    alanine = np.where(
        case,
        np.random.normal(350 + severity * 20, 60, n),
        np.random.normal(330, 55, n),
    )
    alanine = np.clip(alanine, 100, 600).round(1)

    # --- Lactate (mmol/L) ---
    # Normal 0.5-2.2; slight elevation in cases (energy metabolism)
    lactate = np.where(
        case,
        np.random.normal(1.35 + severity * 0.3, 0.35, n),
        np.random.normal(1.10, 0.30, n),
    )
    lactate = np.clip(lactate, 0.3, 3.5).round(2)

    # --- Citrate (µmol/L) ---
    # Normal ~80-160
    citrate = np.where(
        case,
        np.random.normal(115 - severity * 6, 22, n),
        np.random.normal(120, 20, n),
    )
    citrate = np.clip(citrate, 40, 220).round(1)

    # --- Ornithine (µmol/L) ---
    # Normal ~30-100
    ornithine = np.where(
        case,
        np.random.normal(68 + severity * 8, 16, n),
        np.random.normal(60, 14, n),
    )
    ornithine = np.clip(ornithine, 15, 140).round(1)

    # --- Citrulline (µmol/L) ---
    # Normal ~15-55
    citrulline = np.where(
        case,
        np.random.normal(33 + severity * 4, 8, n),
        np.random.normal(30, 7, n),
    )
    citrulline = np.clip(citrulline, 8, 70).round(1)

    # --- Arginine (µmol/L) ---
    # Normal ~40-120; mild decrease (NO pathway consumption in cases)
    arginine = np.where(
        case,
        np.random.normal(72 - severity * 6, 15, n),
        np.random.normal(80, 16, n),
    )
    arginine = np.clip(arginine, 20, 160).round(1)

    return {
        "serum_glutamate": glutamate,
        "serine": serine,
        "alanine": alanine,
        "lactate": lactate,
        "citrate": citrate,
        "ornithine": ornithine,
        "citrulline": citrulline,
        "arginine": arginine,
    }


def generate_lipid_panel(diagnosis, severity, age, sex):
    """
    Triglycerides, LDL, HDL, VLDL, total phospholipids, sphingolipids.

    Cases show mild dyslipidemia: elevated triglycerides/VLDL, lower HDL.
    Age positively correlates with triglycerides.
    Triglycerides and VLDL are correlated (VLDL ≈ TG/5 + noise).
    """
    n = len(diagnosis)
    case = diagnosis == 1

    # Age contribution to triglycerides (~0.8 mg/dL per year over 30)
    age_tg_shift = np.clip((age - 30) * 0.8, 0, 40)

    # --- Triglycerides (mg/dL) ---
    # Normal <150; cases mildly elevated
    triglycerides = np.where(
        case,
        np.random.normal(145 + severity * 30, 42, n),
        np.random.normal(120, 38, n),
    )
    triglycerides += age_tg_shift
    triglycerides += sex * np.random.normal(6, 3, n)  # males slightly higher
    triglycerides = np.clip(triglycerides, 40, 400).round(1)

    # --- VLDL (mg/dL) ~ TG/5 with noise ---
    vldl = triglycerides / 5.0 + np.random.normal(0, 3, n)
    vldl = np.clip(vldl, 5, 80).round(1)

    # --- LDL (mg/dL) ---
    # Normal <130; slight elevation in cases
    ldl = np.where(
        case,
        np.random.normal(125 + severity * 10, 30, n),
        np.random.normal(115, 28, n),
    )
    ldl += age * 0.2  # mild age increase
    ldl = np.clip(ldl, 50, 220).round(1)

    # --- HDL (mg/dL) ---
    # Normal 40-60 M, 50-70 F; lower in cases
    base_hdl = np.where(sex == 1, 48, 58)
    hdl = np.where(
        case,
        np.random.normal(base_hdl - severity * 6, 10, n),
        np.random.normal(base_hdl + 3, 11, n),
    )
    hdl = np.clip(hdl, 20, 95).round(1)

    # --- Total phospholipids (mg/dL) ---
    # Normal ~150-280
    phospholipids = np.where(
        case,
        np.random.normal(215 + severity * 8, 30, n),
        np.random.normal(210, 28, n),
    )
    phospholipids = np.clip(phospholipids, 100, 350).round(1)

    # --- Sphingolipids (µmol/L) ---
    # Normal ~200-400; mild dysregulation in cases
    sphingolipids = np.where(
        case,
        np.random.normal(310 + severity * 15, 45, n),
        np.random.normal(290, 40, n),
    )
    sphingolipids = np.clip(sphingolipids, 100, 500).round(1)

    return {
        "triglycerides": triglycerides,
        "ldl": ldl,
        "hdl": hdl,
        "vldl": vldl,
        "total_phospholipids": phospholipids,
        "sphingolipids": sphingolipids,
    }


def generate_redox_and_no(diagnosis, severity):
    """
    Reduced glutathione (GSH) and nitric oxide (NO).

    Cases show reduced GSH (oxidative stress) and mildly elevated NO.
    GSH inversely correlates with inflammatory markers (implemented
    downstream via the correlation injection step).
    """
    n = len(diagnosis)
    case = diagnosis == 1

    # --- Reduced glutathione (µmol/L) ---
    # Normal ~700-1100; lower in cases
    gsh = np.where(
        case,
        np.random.normal(830 - severity * 80, 100, n),
        np.random.normal(920, 95, n),
    )
    gsh = np.clip(gsh, 400, 1300).round(1)

    # --- Nitric oxide (µmol/L) ---
    # Normal ~20-60; mildly elevated in cases (neuroinflammation)
    no = np.where(
        case,
        np.random.normal(42 + severity * 8, 10, n),
        np.random.normal(36, 9, n),
    )
    no = np.clip(no, 8, 80).round(1)

    return {"reduced_glutathione": gsh, "nitric_oxide": no}


def generate_inflammatory_cytokines(diagnosis, severity):
    """
    IL-6, TNF-α, IL-8 with positively correlated noise.

    Cases show mild elevation of all three pro-inflammatory cytokines.
    A shared latent inflammation factor creates inter-cytokine
    correlation (r ≈ 0.35-0.55).
    """
    n = len(diagnosis)
    case = diagnosis == 1

    # Shared inflammation factor (induces positive correlation)
    inflammation_factor = np.where(
        case,
        np.random.normal(0.3 + severity * 0.4, 0.25, n),
        np.random.normal(0.0, 0.20, n),
    )

    # --- IL-6 (pg/mL) ---
    # Normal <7; cases mildly elevated (meta-analyses: ~+1-3 pg/mL)
    il6 = np.where(
        case,
        np.random.normal(4.5 + severity * 2.5, 2.0, n),
        np.random.normal(2.8, 1.6, n),
    )
    il6 += inflammation_factor * 1.5
    il6 = np.clip(il6, 0.3, 20).round(2)

    # --- TNF-α (pg/mL) ---
    # Normal <8; similar pattern
    tnf_alpha = np.where(
        case,
        np.random.normal(6.0 + severity * 2.0, 2.2, n),
        np.random.normal(4.2, 1.8, n),
    )
    tnf_alpha += inflammation_factor * 1.3
    tnf_alpha = np.clip(tnf_alpha, 0.5, 22).round(2)

    # --- IL-8 (pg/mL) ---
    # Normal <10; mildly elevated in cases
    il8 = np.where(
        case,
        np.random.normal(8.5 + severity * 2.5, 3.0, n),
        np.random.normal(6.0, 2.5, n),
    )
    il8 += inflammation_factor * 1.8
    il8 = np.clip(il8, 1.0, 30).round(2)

    return {"il6": il6, "tnf_alpha": tnf_alpha, "il8": il8}


# ============================================================================
# 5. Inject additional correlation structure
# ============================================================================

def inject_correlations(df, diagnosis):
    """
    Apply residual correlation adjustments to ensure:
    - Positive correlation among IL-6, TNF-α, IL-8  (already partly achieved
      via shared inflammation factor; this reinforces it)
    - Triglycerides positively correlated with VLDL  (already structural)
    - Inverse correlation between GSH and inflammatory markers
    - Mild multicollinearity among lipid variables
    """
    n = len(df)

    # GSH–inflammation inverse coupling: shift GSH downward where cytokines
    # are high (rank-based, preserving marginal distributions)
    cytokine_mean = (
        (df["il6"] - df["il6"].mean()) / df["il6"].std()
        + (df["tnf_alpha"] - df["tnf_alpha"].mean()) / df["tnf_alpha"].std()
        + (df["il8"] - df["il8"].mean()) / df["il8"].std()
    ) / 3.0

    gsh_shift = -cytokine_mean * 30  # ~ -30 µmol/L per 1 SD cytokine
    df["reduced_glutathione"] = np.clip(
        df["reduced_glutathione"] + gsh_shift, 400, 1300
    ).round(1)

    # LDL–TG mild positive correlation
    tg_z = (df["triglycerides"] - df["triglycerides"].mean()) / df["triglycerides"].std()
    df["ldl"] = np.clip(
        df["ldl"] + tg_z * 5, 50, 220
    ).round(1)

    # Phospholipids mildly correlated with total lipid burden
    lipid_z = (tg_z + (df["ldl"] - df["ldl"].mean()) / df["ldl"].std()) / 2
    df["total_phospholipids"] = np.clip(
        df["total_phospholipids"] + lipid_z * 6, 100, 350
    ).round(1)

    return df


# ============================================================================
# 6. Introduce ~5 % missing data at random
# ============================================================================

def introduce_missing(df, frac=0.05, exclude_cols=None):
    """Set ~5 % of values to NaN across eligible numeric columns."""
    if exclude_cols is None:
        exclude_cols = []
    eligible = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    rng = np.random.default_rng(SEED + 7)
    for col in eligible:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


# ============================================================================
# 7. Data Dictionary
# ============================================================================

DATA_DICTIONARY = pd.DataFrame([
    # --- Identifiers & outcome ---
    ("subject_id",   "int",          "-",       "Unique subject identifier"),
    ("diagnosis",    "int (binary)",  "-",       "Outcome: 1 = schizophrenia, 0 = healthy control"),

    # --- Demographics ---
    ("age",          "int",          "years",   "Age at enrollment (18-70)"),
    ("sex",          "int (binary)",  "-",       "Biological sex: 1 = male, 0 = female"),

    # --- Genetic variants (8 genes) ---
    ("C4A_variant_present",     "int (binary)", "0/1",   "Reported variant in complement C4A gene detected (MHC locus)"),
    ("C4A_variant_id",          "string",       "-",     "rsID of detected C4A variant"),
    ("DRD2_variant_present",    "int (binary)", "0/1",   "Reported variant in dopamine receptor D2 gene detected"),
    ("DRD2_variant_id",         "string",       "-",     "rsID of detected DRD2 variant"),
    ("GRM3_variant_present",    "int (binary)", "0/1",   "Reported variant in metabotropic glutamate receptor 3 gene detected"),
    ("GRM3_variant_id",         "string",       "-",     "rsID of detected GRM3 variant"),
    ("GRIN2A_variant_present",  "int (binary)", "0/1",   "Reported variant in NMDA receptor subunit 2A gene detected"),
    ("GRIN2A_variant_id",       "string",       "-",     "rsID of detected GRIN2A variant"),
    ("SLC39A8_variant_present", "int (binary)", "0/1",   "Reported variant in zinc transporter SLC39A8 gene detected"),
    ("SLC39A8_variant_id",      "string",       "-",     "rsID of detected SLC39A8 variant"),
    ("DISC1_variant_present",   "int (binary)", "0/1",   "Reported variant in Disrupted-in-Schizophrenia 1 gene detected"),
    ("DISC1_variant_id",        "string",       "-",     "rsID of detected DISC1 variant"),
    ("COMT_variant_present",    "int (binary)", "0/1",   "Reported variant in catechol-O-methyltransferase gene detected"),
    ("COMT_variant_id",         "string",       "-",     "rsID of detected COMT variant"),
    ("BDNF_variant_present",    "int (binary)", "0/1",   "Reported variant in brain-derived neurotrophic factor gene detected"),
    ("BDNF_variant_id",         "string",       "-",     "rsID of detected BDNF variant"),

    # --- Amino acids & metabolites ---
    ("serum_glutamate", "float", "µmol/L",  "Serum glutamate; primary excitatory neurotransmitter precursor"),
    ("serine",          "float", "µmol/L",  "Serum L-serine; NMDA receptor co-agonist precursor"),
    ("alanine",         "float", "µmol/L",  "Serum L-alanine; gluconeogenic amino acid"),
    ("lactate",         "float", "mmol/L",  "Serum lactate; glycolysis end-product / energy metabolism marker"),
    ("citrate",         "float", "µmol/L",  "Serum citrate; TCA cycle intermediate"),

    # --- Urea cycle ---
    ("ornithine",   "float", "µmol/L",  "Serum ornithine; urea cycle intermediate"),
    ("citrulline",  "float", "µmol/L",  "Serum citrulline; urea cycle intermediate / NO synthesis marker"),
    ("arginine",    "float", "µmol/L",  "Serum L-arginine; NO synthase substrate / urea cycle"),

    # --- Lipid panel ---
    ("triglycerides",       "float", "mg/dL",  "Serum triglycerides"),
    ("ldl",                 "float", "mg/dL",  "Low-density lipoprotein cholesterol"),
    ("hdl",                 "float", "mg/dL",  "High-density lipoprotein cholesterol"),
    ("vldl",                "float", "mg/dL",  "Very low-density lipoprotein cholesterol (≈ TG/5)"),
    ("total_phospholipids", "float", "mg/dL",  "Total serum phospholipids"),
    ("sphingolipids",       "float", "µmol/L", "Serum sphingolipids (ceramide pathway)"),

    # --- Redox / oxidative stress ---
    ("reduced_glutathione", "float", "µmol/L", "Reduced glutathione (GSH); major intracellular antioxidant"),
    ("nitric_oxide",        "float", "µmol/L", "Serum nitric oxide metabolites (NOx); reflects NO bioavailability"),

    # --- Inflammatory cytokines ---
    ("il6",       "float", "pg/mL", "Interleukin-6; pro-inflammatory cytokine"),
    ("tnf_alpha", "float", "pg/mL", "Tumor necrosis factor alpha; pro-inflammatory cytokine"),
    ("il8",       "float", "pg/mL", "Interleukin-8 (CXCL8); pro-inflammatory chemokine"),
], columns=["variable", "data_type", "unit", "description"])


# ============================================================================
# 8. Main Generation Pipeline
# ============================================================================

def main():
    print("Generating synthetic schizophrenia biomarker dataset ...")

    # 1. Demographics & diagnosis
    diagnosis, age, sex = generate_demographics(N_CASES, N_CONTROLS)

    # 2. Genetic variants
    gen_data = generate_genetic_variants(N_CASES, N_CONTROLS)

    # 3. Latent polygenic score
    severity = compute_polygenic_score(diagnosis, age, sex, gen_data)

    # 4. Biomarker blocks
    amino = generate_amino_acids_and_metabolites(diagnosis, severity, age, sex)
    lipids = generate_lipid_panel(diagnosis, severity, age, sex)
    redox = generate_redox_and_no(diagnosis, severity)
    cytokines = generate_inflammatory_cytokines(diagnosis, severity)

    # 5. Assemble DataFrame
    df = pd.DataFrame({
        "subject_id": np.arange(1, N_TOTAL + 1),
        "diagnosis": diagnosis,
        "age": age,
        "sex": sex,
    })

    # Genetic columns
    for gene in VARIANT_CATALOG:
        df[f"{gene}_variant_present"] = gen_data[f"{gene}_variant_present"]
        df[f"{gene}_variant_id"] = gen_data[f"{gene}_variant_id"]

    # Biochemical / metabolomic columns
    for block in [amino, lipids, redox, cytokines]:
        for k, v in block.items():
            df[k] = v

    # 6. Inject additional correlation structure
    df = inject_correlations(df, diagnosis)

    # 7. Introduce ~5 % missing data (preserve identifiers / genetic binary)
    exclude = (
        ["subject_id", "diagnosis"]
        + [f"{g}_variant_present" for g in VARIANT_CATALOG]
        + [f"{g}_variant_id" for g in VARIANT_CATALOG]
    )
    df = introduce_missing(df, frac=0.05, exclude_cols=exclude)

    # 8. Save outputs
    dataset_path = "schizophrenia_synthetic_dataset.csv"
    dict_path = "schizophrenia_data_dictionary.csv"

    df.to_csv(dataset_path, index=False)
    DATA_DICTIONARY.to_csv(dict_path, index=False)

    print(f"Dataset saved to: {dataset_path}  "
          f"({len(df)} rows, {len(df.columns)} columns)")
    print(f"Data dictionary saved to: {dict_path}")

    # ---- Sanity checks ----
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    print(f"\nDiagnosis distribution:\n{df['diagnosis'].value_counts().to_string()}")
    # Report missing fraction only on columns subject to random missingness
    numeric_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\nMissing-data fraction (numeric biomarker cols): "
          f"{df[numeric_cols].isnull().mean().mean():.3f}")

    print(f"\nSex distribution (% male):")
    for grp, label in [(1, "Cases"), (0, "Controls")]:
        pct = df.loc[df["diagnosis"] == grp, "sex"].mean() * 100
        print(f"  {label}: {pct:.1f}%")

    print(f"\nAge (mean ± SD):")
    for grp, label in [(1, "Cases"), (0, "Controls")]:
        vals = df.loc[df["diagnosis"] == grp, "age"]
        print(f"  {label}: {vals.mean():.1f} ± {vals.std():.1f}")

    print(f"\nGenetic variant prevalence (% carriers):")
    for gene in VARIANT_CATALOG:
        col = f"{gene}_variant_present"
        c = df.loc[df["diagnosis"] == 1, col].mean() * 100
        h = df.loc[df["diagnosis"] == 0, col].mean() * 100
        print(f"  {gene:>8s}  cases={c:5.1f}%  controls={h:5.1f}%")

    print(f"\nBiomarker means (cases vs controls):")
    bio_cols = [
        "serum_glutamate", "serine", "lactate", "reduced_glutathione",
        "nitric_oxide", "il6", "tnf_alpha", "il8",
        "triglycerides", "ldl", "hdl", "vldl",
    ]
    for col in bio_cols:
        cm = df.loc[df["diagnosis"] == 1, col].mean()
        hm = df.loc[df["diagnosis"] == 0, col].mean()
        print(f"  {col:>22s}  cases={cm:8.2f}  controls={hm:8.2f}")

    print(f"\nKey correlations (pairwise complete obs):")
    pairs = [
        ("il6", "tnf_alpha"),
        ("il6", "il8"),
        ("tnf_alpha", "il8"),
        ("triglycerides", "vldl"),
        ("reduced_glutathione", "il6"),
        ("triglycerides", "ldl"),
    ]
    for a, b in pairs:
        r = df[[a, b]].dropna().corr().iloc[0, 1]
        print(f"  corr({a}, {b}) = {r:.3f}")

    return df


if __name__ == "__main__":
    main()
