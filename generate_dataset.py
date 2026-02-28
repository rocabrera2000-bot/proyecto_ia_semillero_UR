"""
Synthetic Biomedical Dataset Generator for Hereditary Hemochromatosis

Generates a dataset of 1000 individuals (500 cases, 500 controls) with:
- Genetic variants in iron homeostasis genes (HFE, HJV, HAMP, TFR2, SLC40A1)
- Complete blood count (CBC) parameters
- Biochemical/iron studies
- Derived noninvasive fibrosis indices (APRI, FIB-4, GPR)
- Demographic covariates (age, sex)
- ~5% missing data at random

All values are physiologically plausible with biologically consistent correlations.
"""

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
np.random.seed(SEED)

N_CASES = 500
N_CONTROLS = 500
N_TOTAL = N_CASES + N_CONTROLS

# ---------------------------------------------------------------------------
# 1. Demographics
# ---------------------------------------------------------------------------

def generate_demographics(n_cases, n_controls):
    """Generate age and sex with clinically realistic confounding."""
    # Sex: hemochromatosis more commonly expressed in males
    sex_cases = np.random.binomial(1, 0.65, n_cases)      # 65% male among cases
    sex_controls = np.random.binomial(1, 0.48, n_controls) # 48% male among controls
    sex = np.concatenate([sex_cases, sex_controls])  # 1=male, 0=female

    # Age: cases tend to present in middle age; controls span broader range
    age_cases = np.clip(
        np.random.normal(52, 12, n_cases), 18, 85
    ).astype(int)
    age_controls = np.clip(
        np.random.normal(47, 15, n_controls), 18, 85
    ).astype(int)
    age = np.concatenate([age_cases, age_controls])

    diagnosis = np.concatenate([np.ones(n_cases), np.zeros(n_controls)]).astype(int)
    return diagnosis, age, sex


# ---------------------------------------------------------------------------
# 2. Genetic Variants
# ---------------------------------------------------------------------------

# Known pathogenic / likely pathogenic variants per gene
VARIANT_CATALOG = {
    "HFE": [
        "rs1800562",   # C282Y — most common HH variant
        "rs1799945",   # H63D
        "rs1800730",   # S65C
    ],
    "HJV": [
        "rs74315323",  # G320V — juvenile hemochromatosis
        "rs104894002",  # R385* (nonsense)
    ],
    "HAMP": [
        "rs104894696",  # C78T (hepcidin deficiency)
        "rs373489814",  # R56* — severe juvenile HH
    ],
    "TFR2": [
        "rs121918673",  # Y250X — type 3 hemochromatosis
        "rs199476291",  # AVAQ 594-597del
    ],
    "SLC40A1": [
        "rs11568350",  # N144H — ferroportin disease (type 4)
        "rs104893665",  # A77D
    ],
}

# Carrier frequencies: (prob in cases, prob in controls)
# HFE variants are common; non-HFE genes are rare
GENE_FREQ = {
    "HFE":     (0.72, 0.15),
    "HJV":     (0.08, 0.01),
    "HAMP":    (0.06, 0.008),
    "TFR2":    (0.07, 0.01),
    "SLC40A1": (0.09, 0.012),
}


def generate_genetic_variants(n_cases, n_controls, sex):
    """Generate binary variant presence + specific rsID for each gene."""
    data = {}
    for gene, (freq_case, freq_ctrl) in GENE_FREQ.items():
        presence_cases = np.random.binomial(1, freq_case, n_cases)
        presence_controls = np.random.binomial(1, freq_ctrl, n_controls)
        presence = np.concatenate([presence_cases, presence_controls])

        # Pick a specific variant for each carrier
        variants = VARIANT_CATALOG[gene]
        specific = []
        for p in presence:
            if p == 1:
                specific.append(np.random.choice(variants))
            else:
                specific.append(np.nan)

        data[f"{gene}_variant_present"] = presence
        data[f"{gene}_variant_id"] = specific

    return data


# ---------------------------------------------------------------------------
# 3. Severity / disease stage latent variable
# ---------------------------------------------------------------------------

def compute_severity(diagnosis, age, sex, genetic_data):
    """
    Compute a latent severity score (0-1) for each individual.
    Higher for cases with more genetic burden, older age, and male sex.
    Controls get near-zero severity with some noise.
    """
    n = len(diagnosis)
    severity = np.zeros(n)

    # Genetic burden component
    gene_burden = np.zeros(n)
    gene_weights = {"HFE": 0.45, "HJV": 0.20, "HAMP": 0.15,
                    "TFR2": 0.12, "SLC40A1": 0.08}
    for gene, w in gene_weights.items():
        gene_burden += genetic_data[f"{gene}_variant_present"] * w

    # Severity for cases — moderate with high variance for overlap
    case_mask = diagnosis == 1
    severity[case_mask] = (
        0.18
        + 0.22 * gene_burden[case_mask]
        + 0.08 * ((age[case_mask] - 18) / 67)
        + 0.06 * sex[case_mask]
        + np.random.normal(0, 0.12, case_mask.sum())
    )
    # Severity for controls — low but with some upward noise
    ctrl_mask = diagnosis == 0
    severity[ctrl_mask] = (
        0.05
        + 0.08 * gene_burden[ctrl_mask]
        + 0.02 * ((age[ctrl_mask] - 18) / 67)
        + np.random.normal(0, 0.06, ctrl_mask.sum())
    )
    severity = np.clip(severity, 0, 1)
    return severity


# ---------------------------------------------------------------------------
# 4. Iron Studies & Biochemistry
# ---------------------------------------------------------------------------

def generate_iron_studies(diagnosis, severity, sex):
    """Generate iron panel with biologically plausible correlations."""
    n = len(diagnosis)
    case_mask = diagnosis == 1

    # --- Serum iron (µg/dL) ---
    # Normal: 60-170; elevated in overload but with substantial overlap
    serum_iron = np.where(
        case_mask,
        np.random.normal(130 + severity * 40, 45, n),
        np.random.normal(100, 32, n),
    )
    # Males tend slightly higher
    serum_iron += sex * np.random.normal(8, 3, n)
    serum_iron = np.clip(serum_iron, 30, 350)

    # --- TIBC (µg/dL) ---
    # Normal: 250-370; tends lower in iron overload but with overlap
    tibc = np.where(
        case_mask,
        np.random.normal(290 - severity * 30, 40, n),
        np.random.normal(320, 38, n),
    )
    tibc = np.clip(tibc, 180, 450)

    # --- UIBC (µg/dL) = TIBC - serum_iron ---
    uibc = tibc - serum_iron
    uibc = np.clip(uibc, 0, 350)

    # --- Transferrin saturation (%) = serum_iron / TIBC * 100 ---
    tsat = (serum_iron / tibc) * 100
    tsat = np.clip(tsat, 5, 100)

    # --- Ferritin (ng/mL) ---
    # Normal male: 20-250; female: 10-120; elevated in overload with
    # wide spread to create meaningful overlap between groups
    base_ferritin = np.where(sex == 1, 120, 55)
    ferritin = np.where(
        case_mask,
        np.random.lognormal(
            np.log(base_ferritin + severity * 400 + 40), 0.55, n
        ),
        np.random.lognormal(np.log(base_ferritin + 15), 0.50, n),
    )
    # Allow a few very high ferritins in severe cases (clinically plausible)
    extreme_mask = case_mask & (severity > 0.80)
    ferritin[extreme_mask] *= np.random.uniform(1.1, 1.8, extreme_mask.sum())
    ferritin = np.clip(ferritin, 5, 5000)

    return serum_iron, tibc, uibc, tsat, ferritin


# ---------------------------------------------------------------------------
# 5. Liver enzymes
# ---------------------------------------------------------------------------

def generate_liver_enzymes(diagnosis, severity, sex, age):
    """AST, ALT, GGT with hepatic involvement in subset of cases."""
    n = len(diagnosis)
    case_mask = diagnosis == 1

    # Hepatic involvement probability increases with severity (subset only)
    hepatic = np.zeros(n, dtype=bool)
    hepatic[case_mask] = np.random.random(case_mask.sum()) < (0.2 + 0.35 * severity[case_mask])

    # --- AST (U/L) ---
    ast = np.where(
        hepatic,
        np.random.normal(42 + severity * 35, 15, n),
        np.random.normal(25, 8, n),
    )
    ast += sex * np.random.normal(3, 1.5, n)  # slightly higher in males
    ast = np.clip(ast, 8, 200).round(1)

    # --- ALT (U/L) ---
    # Correlated with AST
    alt = ast * np.random.normal(1.1, 0.15, n)
    alt = np.where(
        hepatic,
        alt + np.random.normal(5, 6, n),
        np.random.normal(23, 9, n),
    )
    alt = np.clip(alt, 5, 220).round(1)

    # --- GGT (U/L) ---
    ggt = np.where(
        hepatic,
        np.random.normal(50 + severity * 50, 22, n),
        np.random.normal(30, 14, n),
    )
    ggt += sex * np.random.normal(8, 3, n)
    ggt += (age - 40) * 0.15  # age-related increase
    ggt = np.clip(ggt, 5, 350).round(1)

    return ast, alt, ggt


# ---------------------------------------------------------------------------
# 6. Complete Blood Count (CBC)
# ---------------------------------------------------------------------------

def generate_cbc(diagnosis, severity, sex, age):
    """Generate CBC with mild thrombocytopenia in advanced disease."""
    n = len(diagnosis)
    case_mask = diagnosis == 1

    # --- Hemoglobin (g/dL) ---
    hgb = np.where(
        sex == 1,
        np.random.normal(15.0, 1.1, n),
        np.random.normal(13.2, 1.0, n),
    )
    # Slight increase in iron overload
    hgb[case_mask] += np.random.normal(0.4, 0.3, case_mask.sum())
    hgb = np.clip(hgb, 9.0, 19.0).round(1)

    # --- Hematocrit (%) — roughly Hgb * 3 with noise ---
    hct = hgb * np.random.normal(3.0, 0.08, n)
    hct = np.clip(hct, 28.0, 56.0).round(1)

    # --- RBC count (×10^12/L) ---
    rbc = np.where(
        sex == 1,
        np.random.normal(5.1, 0.4, n),
        np.random.normal(4.5, 0.35, n),
    )
    rbc = np.clip(rbc, 3.0, 6.5).round(2)

    # --- MCV (fL) ---
    # Calculated roughly from Hct/RBC * 10
    mcv = (hct / rbc) * 10 + np.random.normal(0, 1.5, n)
    mcv = np.clip(mcv, 70, 105).round(1)

    # --- MCH (pg) ---
    mch = (hgb / rbc) * 10 + np.random.normal(0, 0.8, n)
    mch = np.clip(mch, 22, 36).round(1)

    # --- MCHC (g/dL) ---
    mchc = (hgb / hct) * 100 + np.random.normal(0, 0.5, n)
    mchc = np.clip(mchc, 30, 37).round(1)

    # --- RDW (%) ---
    rdw = np.random.normal(13.5, 1.2, n)
    rdw[case_mask] += np.random.normal(0.3, 0.4, case_mask.sum())
    rdw = np.clip(rdw, 11.0, 20.0).round(1)

    # --- WBC (×10^9/L) ---
    wbc = np.random.normal(6.8, 1.8, n)
    wbc = np.clip(wbc, 2.5, 15.0).round(1)

    # WBC differential (fractions that sum to ~1)
    neutrophils_pct = np.random.normal(58, 7, n)
    lymphocytes_pct = np.random.normal(30, 6, n)
    monocytes_pct = np.random.normal(6, 1.5, n)
    eosinophils_pct = np.random.normal(3, 1.2, n)
    basophils_pct = np.random.normal(0.8, 0.4, n)

    # Normalize to 100%
    total_pct = (neutrophils_pct + lymphocytes_pct + monocytes_pct
                 + eosinophils_pct + basophils_pct)
    neutrophils_pct = (neutrophils_pct / total_pct * 100).round(1)
    lymphocytes_pct = (lymphocytes_pct / total_pct * 100).round(1)
    monocytes_pct = (monocytes_pct / total_pct * 100).round(1)
    eosinophils_pct = (eosinophils_pct / total_pct * 100).round(1)
    basophils_pct = (100 - neutrophils_pct - lymphocytes_pct
                     - monocytes_pct - eosinophils_pct).round(1)

    neutrophils_pct = np.clip(neutrophils_pct, 35, 80)
    lymphocytes_pct = np.clip(lymphocytes_pct, 15, 50)
    monocytes_pct = np.clip(monocytes_pct, 2, 12)
    eosinophils_pct = np.clip(eosinophils_pct, 0.5, 8)
    basophils_pct = np.clip(basophils_pct, 0.0, 3.0)

    # Absolute counts (×10^9/L)
    neutrophils_abs = (wbc * neutrophils_pct / 100).round(2)
    lymphocytes_abs = (wbc * lymphocytes_pct / 100).round(2)
    monocytes_abs = (wbc * monocytes_pct / 100).round(2)
    eosinophils_abs = (wbc * eosinophils_pct / 100).round(2)
    basophils_abs = (wbc * basophils_pct / 100).round(2)

    # --- Platelet count (×10^9/L) ---
    platelets = np.random.normal(245, 55, n)
    # Mild thrombocytopenia in advanced disease (severity > 0.65 in cases)
    advanced_mask = (diagnosis == 1) & (severity > 0.65)
    platelets[advanced_mask] -= np.random.normal(50, 20, advanced_mask.sum())
    platelets = np.clip(platelets, 80, 450).round(0).astype(int)

    return {
        "hemoglobin": hgb,
        "hematocrit": hct,
        "rbc_count": rbc,
        "mcv": mcv,
        "mch": mch,
        "mchc": mchc,
        "rdw": rdw,
        "wbc_count": wbc,
        "neutrophils_pct": neutrophils_pct,
        "lymphocytes_pct": lymphocytes_pct,
        "monocytes_pct": monocytes_pct,
        "eosinophils_pct": eosinophils_pct,
        "basophils_pct": basophils_pct,
        "neutrophils_abs": neutrophils_abs,
        "lymphocytes_abs": lymphocytes_abs,
        "monocytes_abs": monocytes_abs,
        "eosinophils_abs": eosinophils_abs,
        "basophils_abs": basophils_abs,
        "platelet_count": platelets,
    }


# ---------------------------------------------------------------------------
# 7. Derived Fibrosis Indices (mathematically consistent)
# ---------------------------------------------------------------------------

def compute_fibrosis_indices(ast, platelets, alt, age):
    """
    APRI  = (AST / upper_limit_normal_AST) / platelet_count(10^9/L) * 100
    FIB-4 = (age × AST) / (platelet_count(10^9/L) × sqrt(ALT))
    GPR   = (GGT / upper_limit_normal_GGT) / platelet_count(10^9/L) * 100
    Note: GPR is computed externally since GGT is not passed here.
    """
    uln_ast = 40.0  # upper limit of normal for AST

    apri = ((ast / uln_ast) / platelets) * 100
    apri = np.round(apri, 3)

    fib4 = (age * ast) / (platelets * np.sqrt(alt))
    fib4 = np.round(fib4, 3)

    return apri, fib4


def compute_gpr(ggt, platelets):
    """GPR = (GGT / ULN_GGT) / platelet_count(10^9/L) * 100."""
    uln_ggt = 60.0  # upper limit of normal for GGT
    gpr = ((ggt / uln_ggt) / platelets) * 100
    return np.round(gpr, 3)


# ---------------------------------------------------------------------------
# 8. Introduce ~5% missing data at random
# ---------------------------------------------------------------------------

def introduce_missing(df, frac=0.05, exclude_cols=None):
    """Set ~5% of values to NaN across eligible columns."""
    if exclude_cols is None:
        exclude_cols = []
    eligible = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in (np.float64, np.int64, "float64", "int64",
                                     np.int32, np.float32)]
    rng = np.random.default_rng(SEED + 7)
    for col in eligible:
        mask = rng.random(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 9. Data dictionary
# ---------------------------------------------------------------------------

DATA_DICTIONARY = pd.DataFrame([
    ("subject_id", "int", "Unique subject identifier", "-"),
    ("diagnosis", "int (binary)", "Outcome: 1 = hemochromatosis, 0 = control", "-"),
    ("age", "int", "Age at enrollment", "years"),
    ("sex", "int (binary)", "Biological sex: 1 = male, 0 = female", "-"),

    ("HFE_variant_present", "int (binary)", "Pathogenic/likely pathogenic HFE variant detected", "0/1"),
    ("HFE_variant_id", "string", "rsID of detected HFE variant (e.g., rs1800562 C282Y)", "-"),
    ("HJV_variant_present", "int (binary)", "Pathogenic/likely pathogenic HJV variant detected", "0/1"),
    ("HJV_variant_id", "string", "rsID of detected HJV variant", "-"),
    ("HAMP_variant_present", "int (binary)", "Pathogenic/likely pathogenic HAMP variant detected", "0/1"),
    ("HAMP_variant_id", "string", "rsID of detected HAMP variant", "-"),
    ("TFR2_variant_present", "int (binary)", "Pathogenic/likely pathogenic TFR2 variant detected", "0/1"),
    ("TFR2_variant_id", "string", "rsID of detected TFR2 variant", "-"),
    ("SLC40A1_variant_present", "int (binary)", "Pathogenic/likely pathogenic SLC40A1 variant detected", "0/1"),
    ("SLC40A1_variant_id", "string", "rsID of detected SLC40A1 variant", "-"),

    ("hemoglobin", "float", "Hemoglobin concentration", "g/dL"),
    ("hematocrit", "float", "Hematocrit (packed cell volume)", "%"),
    ("rbc_count", "float", "Red blood cell count", "×10^12/L"),
    ("mcv", "float", "Mean corpuscular volume", "fL"),
    ("mch", "float", "Mean corpuscular hemoglobin", "pg"),
    ("mchc", "float", "Mean corpuscular hemoglobin concentration", "g/dL"),
    ("rdw", "float", "Red cell distribution width", "%"),
    ("wbc_count", "float", "White blood cell count", "×10^9/L"),
    ("neutrophils_pct", "float", "Neutrophils (percentage of WBC)", "%"),
    ("lymphocytes_pct", "float", "Lymphocytes (percentage of WBC)", "%"),
    ("monocytes_pct", "float", "Monocytes (percentage of WBC)", "%"),
    ("eosinophils_pct", "float", "Eosinophils (percentage of WBC)", "%"),
    ("basophils_pct", "float", "Basophils (percentage of WBC)", "%"),
    ("neutrophils_abs", "float", "Neutrophils (absolute count)", "×10^9/L"),
    ("lymphocytes_abs", "float", "Lymphocytes (absolute count)", "×10^9/L"),
    ("monocytes_abs", "float", "Monocytes (absolute count)", "×10^9/L"),
    ("eosinophils_abs", "float", "Eosinophils (absolute count)", "×10^9/L"),
    ("basophils_abs", "float", "Basophils (absolute count)", "×10^9/L"),
    ("platelet_count", "int", "Platelet count", "×10^9/L"),

    ("serum_iron", "float", "Serum iron", "µg/dL"),
    ("ferritin", "float", "Serum ferritin", "ng/mL"),
    ("tibc", "float", "Total iron-binding capacity", "µg/dL"),
    ("uibc", "float", "Unsaturated iron-binding capacity (TIBC − serum iron)", "µg/dL"),
    ("transferrin_saturation", "float", "Transferrin saturation (serum iron / TIBC × 100)", "%"),

    ("ast", "float", "Aspartate aminotransferase", "U/L"),
    ("alt", "float", "Alanine aminotransferase", "U/L"),
    ("ggt", "float", "Gamma-glutamyl transferase", "U/L"),

    ("apri", "float", "AST-to-Platelet Ratio Index: (AST/ULN_AST)/platelets×100; ULN_AST=40", "-"),
    ("fib4", "float", "Fibrosis-4 Index: (age×AST)/(platelets×√ALT)", "-"),
    ("gpr", "float", "GGT-to-Platelet Ratio: (GGT/ULN_GGT)/platelets×100; ULN_GGT=60", "-"),
], columns=["variable", "data_type", "description", "unit"])


# ---------------------------------------------------------------------------
# 10. Main generation pipeline
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic hemochromatosis dataset ...")

    # Demographics & diagnosis
    diagnosis, age, sex = generate_demographics(N_CASES, N_CONTROLS)

    # Genetic variants
    gen_data = generate_genetic_variants(N_CASES, N_CONTROLS, sex)

    # Latent severity
    severity = compute_severity(diagnosis, age, sex, gen_data)

    # Iron studies
    serum_iron, tibc, uibc, tsat, ferritin = generate_iron_studies(
        diagnosis, severity, sex
    )

    # Liver enzymes
    ast, alt, ggt = generate_liver_enzymes(diagnosis, severity, sex, age)

    # CBC
    cbc = generate_cbc(diagnosis, severity, sex, age)

    # Fibrosis indices (computed from raw values for consistency)
    apri, fib4 = compute_fibrosis_indices(
        ast, cbc["platelet_count"].astype(float), alt, age.astype(float)
    )
    gpr = compute_gpr(ggt, cbc["platelet_count"].astype(float))

    # Assemble DataFrame
    df = pd.DataFrame({
        "subject_id": np.arange(1, N_TOTAL + 1),
        "diagnosis": diagnosis,
        "age": age,
        "sex": sex,
    })

    # Genetic columns
    for gene in ["HFE", "HJV", "HAMP", "TFR2", "SLC40A1"]:
        df[f"{gene}_variant_present"] = gen_data[f"{gene}_variant_present"]
        df[f"{gene}_variant_id"] = gen_data[f"{gene}_variant_id"]

    # CBC
    for k, v in cbc.items():
        df[k] = v

    # Iron panel
    df["serum_iron"] = np.round(serum_iron, 1)
    df["ferritin"] = np.round(ferritin, 1)
    df["tibc"] = np.round(tibc, 1)
    df["uibc"] = np.round(uibc, 1)
    df["transferrin_saturation"] = np.round(tsat, 1)

    # Liver enzymes
    df["ast"] = ast
    df["alt"] = alt
    df["ggt"] = ggt

    # Fibrosis indices
    df["apri"] = apri
    df["fib4"] = fib4
    df["gpr"] = gpr

    # Introduce ~5% missing data (exclude identifiers and string columns)
    exclude = ["subject_id", "diagnosis",
               "HFE_variant_present", "HJV_variant_present",
               "HAMP_variant_present", "TFR2_variant_present",
               "SLC40A1_variant_present",
               "HFE_variant_id", "HJV_variant_id", "HAMP_variant_id",
               "TFR2_variant_id", "SLC40A1_variant_id"]
    df = introduce_missing(df, frac=0.05, exclude_cols=exclude)

    # Save outputs
    dataset_path = "hemochromatosis_synthetic_dataset.csv"
    dict_path = "data_dictionary.csv"

    df.to_csv(dataset_path, index=False)
    DATA_DICTIONARY.to_csv(dict_path, index=False)

    print(f"Dataset saved to: {dataset_path}  ({len(df)} rows, {len(df.columns)} columns)")
    print(f"Data dictionary saved to: {dict_path}")

    # Quick summary statistics
    print("\n--- Quick Sanity Checks ---")
    print(f"Diagnosis distribution:\n{df['diagnosis'].value_counts().to_string()}")
    print(f"\nMissing data fraction: {df.isnull().mean().mean():.3f}")
    print(f"\nIron panel means by diagnosis:")
    iron_cols = ["serum_iron", "ferritin", "tibc", "transferrin_saturation"]
    for col in iron_cols:
        case_mean = df.loc[df["diagnosis"] == 1, col].mean()
        ctrl_mean = df.loc[df["diagnosis"] == 0, col].mean()
        print(f"  {col:>25s}  case={case_mean:8.1f}  control={ctrl_mean:8.1f}")

    print(f"\nFibrosis index means by diagnosis:")
    for col in ["apri", "fib4", "gpr"]:
        case_mean = df.loc[df["diagnosis"] == 1, col].mean()
        ctrl_mean = df.loc[df["diagnosis"] == 0, col].mean()
        print(f"  {col:>6s}  case={case_mean:8.3f}  control={ctrl_mean:8.3f}")

    print(f"\nGenetic variant prevalence (% carriers):")
    for gene in ["HFE", "HJV", "HAMP", "TFR2", "SLC40A1"]:
        col = f"{gene}_variant_present"
        case_pct = df.loc[df["diagnosis"] == 1, col].mean() * 100
        ctrl_pct = df.loc[df["diagnosis"] == 0, col].mean() * 100
        print(f"  {gene:>8s}  case={case_pct:5.1f}%  control={ctrl_pct:5.1f}%")

    # Verify mathematical consistency of derived indices (on non-missing rows)
    check = df.dropna(subset=["ast", "platelet_count", "alt", "age", "ggt",
                               "apri", "fib4", "gpr"])
    recomputed_apri = ((check["ast"] / 40.0) / check["platelet_count"]) * 100
    recomputed_fib4 = (check["age"] * check["ast"]) / (
        check["platelet_count"] * np.sqrt(check["alt"])
    )
    recomputed_gpr = ((check["ggt"] / 60.0) / check["platelet_count"]) * 100

    apri_ok = np.allclose(check["apri"], recomputed_apri, atol=0.01)
    fib4_ok = np.allclose(check["fib4"], recomputed_fib4, atol=0.01)
    gpr_ok = np.allclose(check["gpr"], recomputed_gpr, atol=0.01)

    print(f"\nDerived index consistency checks:")
    print(f"  APRI internally consistent: {apri_ok}")
    print(f"  FIB-4 internally consistent: {fib4_ok}")
    print(f"  GPR  internally consistent: {gpr_ok}")

    return df


if __name__ == "__main__":
    main()
