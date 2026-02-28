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

DESIGN NOTES — Realistic difficulty + ML advantage:
  Linear (main-effect) signal is deliberately WEAK (Cohen's d ≈ 0.10-0.20
  per biomarker) so logistic regression achieves only moderate AUC.

  Additional signal is embedded directly as NONLINEAR INTERACTIONS in the
  observable features — NOT funneled through a smooth latent variable:

    • Gene-gene epistasis drives biomarker subgroups:
      - C4A × COMT carriers (cases) → strong glutamate spike + GSH drop
      - DRD2 × BDNF carriers (cases) → elevated cytokines
      - GRIN2A × GRM3 carriers (cases) → lactate + lipid shift
    • Threshold effects:
      - High glutamate + low GSH (cases) → extra push
      - High TG + low HDL (cases) → extra push
    • Age × gene interaction:
      - Young (<30) COMT+ cases → amplified inflammatory signal

  These create distinct subpopulations where COMBINATIONS of features are
  shifted, but marginal distributions heavily overlap. Trees/SVMs/NNs can
  learn these interaction rules; logistic regression (on raw features) cannot.

  Target performance:
    Logistic regression:  AUC ≈ 0.70-0.76
    Best ML model:        AUC ≈ 0.82-0.88
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
    """Generate age and sex with mild confounding."""
    sex_cases = np.random.binomial(1, 0.58, n_cases)
    sex_controls = np.random.binomial(1, 0.48, n_controls)
    sex = np.concatenate([sex_cases, sex_controls])

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
# 2. Genetic Variants
# ============================================================================

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

# (freq_in_cases, freq_in_controls)
GENE_PARAMS = {
    "C4A":     (0.30, 0.22),
    "DRD2":    (0.25, 0.20),
    "GRM3":    (0.20, 0.16),
    "GRIN2A":  (0.16, 0.12),
    "SLC39A8": (0.09, 0.07),
    "DISC1":   (0.13, 0.10),
    "COMT":    (0.38, 0.32),
    "BDNF":    (0.28, 0.22),
}


def generate_genetic_variants(n_cases, n_controls):
    """Generate binary variant presence + specific rsID per gene."""
    data = {}
    for gene, (freq_case, freq_ctrl) in GENE_PARAMS.items():
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
# 3. Baseline biomarkers (NEAR-IDENTICAL distributions for cases/controls)
#    The tiny main effects alone give logistic regression minimal power.
# ============================================================================

def generate_baseline_biomarkers(n, age, sex):
    """
    Generate biomarkers from SHARED population distributions.
    No case/control separation at this stage — both groups drawn from
    the same physiological range with identical means.
    Age and sex confounders are applied.
    """
    data = {}

    # --- Amino acids & metabolites ---
    data["serum_glutamate"] = np.clip(
        np.random.normal(50, 15, n), 10, 120
    ).round(1)

    data["serine"] = np.clip(
        np.random.normal(112, 22, n), 40, 200
    ).round(1)

    data["alanine"] = np.clip(
        np.random.normal(335, 62, n), 100, 600
    ).round(1)

    data["lactate"] = np.clip(
        np.random.normal(1.15, 0.36, n), 0.3, 3.5
    ).round(2)

    data["citrate"] = np.clip(
        np.random.normal(118, 23, n), 40, 220
    ).round(1)

    data["ornithine"] = np.clip(
        np.random.normal(61, 16, n), 15, 140
    ).round(1)

    data["citrulline"] = np.clip(
        np.random.normal(30, 8, n), 8, 70
    ).round(1)

    data["arginine"] = np.clip(
        np.random.normal(78, 17, n), 20, 160
    ).round(1)

    # --- Lipids (with age/sex confounding) ---
    age_tg = np.clip((age - 30) * 0.8, 0, 40)
    data["triglycerides"] = np.clip(
        np.random.normal(128, 46, n) + age_tg + sex * 6,
        40, 400
    ).round(1)

    data["vldl"] = np.clip(
        data["triglycerides"] / 5.0 + np.random.normal(0, 3.5, n),
        5, 80
    ).round(1)

    data["ldl"] = np.clip(
        np.random.normal(118, 31, n) + age * 0.2,
        50, 220
    ).round(1)

    base_hdl = np.where(sex == 1, 48.0, 58.0)
    data["hdl"] = np.clip(
        np.random.normal(base_hdl, 12, n),
        20, 95
    ).round(1)

    data["total_phospholipids"] = np.clip(
        np.random.normal(211, 31, n), 100, 350
    ).round(1)

    data["sphingolipids"] = np.clip(
        np.random.normal(295, 47, n), 100, 500
    ).round(1)

    # --- Redox ---
    data["reduced_glutathione"] = np.clip(
        np.random.normal(910, 102, n), 400, 1300
    ).round(1)

    data["nitric_oxide"] = np.clip(
        np.random.normal(37, 10, n), 8, 80
    ).round(1)

    # --- Cytokines ---
    # Shared latent inflammation factor (population-wide)
    infl = np.random.normal(0, 0.22, n)
    data["il6"] = np.clip(
        np.random.normal(3.0, 1.9, n) + infl * 1.2, 0.3, 20
    ).round(2)
    data["tnf_alpha"] = np.clip(
        np.random.normal(4.4, 2.1, n) + infl * 1.0, 0.5, 22
    ).round(2)
    data["il8"] = np.clip(
        np.random.normal(6.5, 2.9, n) + infl * 1.4, 1.0, 30
    ).round(2)

    return data


# ============================================================================
# 4. Apply WEAK linear main effects (what LR can catch — small)
# ============================================================================

def apply_weak_main_effects(data, diagnosis):
    """
    Apply WEAK main-effect shifts to case biomarkers (d ≈ 0.08-0.15).
    This gives LR a modest baseline, but the bulk of the signal comes
    from the interaction layer.
    """
    case = diagnosis == 1
    nc = case.sum()

    data["serum_glutamate"][case] += np.random.normal(1.5, 1.0, nc)
    data["serine"][case] -= np.random.normal(1.5, 1.0, nc)
    data["lactate"][case] += np.random.normal(0.03, 0.02, nc)
    data["reduced_glutathione"][case] -= np.random.normal(10, 5, nc)
    data["nitric_oxide"][case] += np.random.normal(1.5, 0.8, nc)
    data["il6"][case] += np.random.normal(0.25, 0.15, nc)
    data["tnf_alpha"][case] += np.random.normal(0.2, 0.12, nc)
    data["il8"][case] += np.random.normal(0.25, 0.15, nc)
    data["triglycerides"][case] += np.random.normal(4, 2.5, nc)
    data["hdl"][case] -= np.random.normal(1.2, 0.8, nc)
    data["ornithine"][case] += np.random.normal(1.0, 0.6, nc)
    data["arginine"][case] -= np.random.normal(1.0, 0.6, nc)

    return data


# ============================================================================
# 5. Apply STRONG nonlinear interaction effects
#    These are conditional: they ONLY activate in specific gene × gene or
#    gene × demographic subgroups among cases.  Controls with the same
#    genotypes do NOT get the shift (disease-conditional epistasis).
# ============================================================================

def apply_nonlinear_interactions(data, diagnosis, age, sex, genetic_data):
    """
    Embed XOR-like interaction patterns that create a genuine ML advantage.

    CORE MECHANISM — Bidirectional conditional effects:
      The DIRECTION of a biomarker shift reverses based on genetic context.
      This creates strong signal in the joint distribution but near-zero
      signal in the marginals, because opposite shifts cancel out.

      Example: COMT+ cases have HIGH glutamate / LOW GSH,
               COMT- cases have LOW glutamate / HIGH GSH (protective shift).
               Marginal glutamate ≈ same for cases and controls.
               But trees learn: "if COMT=1 & glutamate>58 → case"
                            and "if COMT=0 & glutamate<42 → case"
               LR with a single coefficient for glutamate sees ~zero signal.

    Three XOR axes, each controlled by a different gene:
      1. COMT  → glutamate / GSH axis
      2. BDNF  → cytokine / nitric oxide axis
      3. DRD2  → lipid (TG, HDL) axis

    Epistatic amplifiers (2-gene) stack additional signal.
    Controls get tiny symmetric noise (no net shift).
    """
    case = diagnosis == 1
    ctrl = diagnosis == 0
    n = len(diagnosis)

    comt = genetic_data["COMT_variant_present"].astype(bool)
    bdnf = genetic_data["BDNF_variant_present"].astype(bool)
    drd2 = genetic_data["DRD2_variant_present"].astype(bool)
    c4a = genetic_data["C4A_variant_present"].astype(bool)
    grin2a = genetic_data["GRIN2A_variant_present"].astype(bool)
    grm3 = genetic_data["GRM3_variant_present"].astype(bool)
    disc1 = genetic_data["DISC1_variant_present"].astype(bool)

    # ================================================================
    # XOR AXIS 1: COMT × glutamate/GSH
    # COMT+ cases: glutamate ↑, GSH ↓
    # COMT- cases: glutamate ↓, GSH ↑ (reversal!)
    # Net marginal ≈ 0 → invisible to LR
    # ================================================================
    mask_comt_case = case & comt           # ~37% of cases
    mask_nocomt_case = case & ~comt        # ~63% of cases
    nc1 = mask_comt_case.sum()
    nc2 = mask_nocomt_case.sum()

    if nc1 > 0:
        data["serum_glutamate"][mask_comt_case] += np.random.normal(14, 3.5, nc1)
        data["reduced_glutathione"][mask_comt_case] -= np.random.normal(80, 18, nc1)
        data["serine"][mask_comt_case] -= np.random.normal(10, 3, nc1)
        data["lactate"][mask_comt_case] += np.random.normal(0.15, 0.05, nc1)

    if nc2 > 0:
        data["serum_glutamate"][mask_nocomt_case] -= np.random.normal(7, 2.5, nc2)
        data["reduced_glutathione"][mask_nocomt_case] += np.random.normal(25, 10, nc2)
        data["serine"][mask_nocomt_case] += np.random.normal(5, 2, nc2)

    # ================================================================
    # XOR AXIS 2: BDNF × cytokine/NO axis
    # BDNF+ cases: IL-6 ↑, TNF-α ↑, IL-8 ↑, NO ↑
    # BDNF- cases: IL-6 ↓ mildly (or flat), TNF-α ↓
    # ================================================================
    mask_bdnf_case = case & bdnf           # ~30%
    mask_nobdnf_case = case & ~bdnf        # ~70%
    nb1 = mask_bdnf_case.sum()
    nb2 = mask_nobdnf_case.sum()

    if nb1 > 0:
        data["il6"][mask_bdnf_case] += np.random.normal(3.0, 0.7, nb1)
        data["tnf_alpha"][mask_bdnf_case] += np.random.normal(2.5, 0.6, nb1)
        data["il8"][mask_bdnf_case] += np.random.normal(3.0, 0.7, nb1)
        data["nitric_oxide"][mask_bdnf_case] += np.random.normal(6, 1.8, nb1)

    if nb2 > 0:
        data["il6"][mask_nobdnf_case] -= np.random.normal(1.0, 0.4, nb2)
        data["tnf_alpha"][mask_nobdnf_case] -= np.random.normal(0.8, 0.3, nb2)
        data["il8"][mask_nobdnf_case] -= np.random.normal(0.8, 0.35, nb2)

    # ================================================================
    # XOR AXIS 3: DRD2 × lipid axis
    # DRD2+ cases: TG ↑↑, HDL ↓↓
    # DRD2- cases: TG ↓ mildly, HDL ↑ mildly
    # ================================================================
    mask_drd2_case = case & drd2           # ~26%
    mask_nodrd2_case = case & ~drd2        # ~74%
    nd1 = mask_drd2_case.sum()
    nd2 = mask_nodrd2_case.sum()

    if nd1 > 0:
        data["triglycerides"][mask_drd2_case] += np.random.normal(38, 8, nd1)
        data["hdl"][mask_drd2_case] -= np.random.normal(9, 2.5, nd1)
        data["sphingolipids"][mask_drd2_case] += np.random.normal(22, 7, nd1)

    if nd2 > 0:
        data["triglycerides"][mask_nodrd2_case] -= np.random.normal(8, 3.5, nd2)
        data["hdl"][mask_nodrd2_case] += np.random.normal(3, 1.5, nd2)

    # ================================================================
    # Epistatic amplifiers (stacked, same direction as parent axis)
    # ================================================================

    # C4A × COMT cases → amplified glutamate/GSH crash
    mask_ep1 = case & c4a & comt
    n1 = mask_ep1.sum()
    if n1 > 0:
        data["serum_glutamate"][mask_ep1] += np.random.normal(5, 2, n1)
        data["reduced_glutathione"][mask_ep1] -= np.random.normal(25, 8, n1)

    # DRD2 × BDNF cases → amplified cytokines
    mask_ep2 = case & drd2 & bdnf
    n2 = mask_ep2.sum()
    if n2 > 0:
        data["il6"][mask_ep2] += np.random.normal(1.5, 0.4, n2)
        data["il8"][mask_ep2] += np.random.normal(1.5, 0.5, n2)

    # GRIN2A × GRM3 cases → glutamate receptor synergy
    mask_ep3 = case & grin2a & grm3
    n3 = mask_ep3.sum()
    if n3 > 0:
        data["serum_glutamate"][mask_ep3] += np.random.normal(5, 2, n3)
        data["lactate"][mask_ep3] += np.random.normal(0.12, 0.04, n3)

    # Young (<30) × gene-carrier cases → age-dependent amplification
    young = age < 30
    mask_yg = case & young & (comt | bdnf | drd2)
    nyg = mask_yg.sum()
    if nyg > 0:
        data["il6"][mask_yg] += np.random.normal(0.5, 0.2, nyg)
        data["reduced_glutathione"][mask_yg] -= np.random.normal(15, 5, nyg)

    # ================================================================
    # Controls: symmetric ghost noise (no net shift, prevents shortcut)
    # ================================================================
    for mask_ctrl, shifts in [
        (ctrl & comt, [("serum_glutamate", 0.5, 2.0),
                        ("reduced_glutathione", -3, 6)]),
        (ctrl & ~comt, [("serum_glutamate", -0.3, 1.5),
                         ("reduced_glutathione", 2, 5)]),
        (ctrl & bdnf, [("il6", 0.1, 0.3),
                        ("tnf_alpha", 0.08, 0.2)]),
        (ctrl & drd2, [("triglycerides", 1.5, 3),
                        ("hdl", -0.3, 1.2)]),
    ]:
        nm = mask_ctrl.sum()
        if nm > 0:
            for col, mu, sd in shifts:
                data[col][mask_ctrl] += np.random.normal(mu, sd, nm)

    return data


# ============================================================================
# 6. Post-hoc correlation tuning
# ============================================================================

def tune_correlations(data, n):
    """
    Reinforce biologically expected correlations without adding
    case/control signal.
    """
    # GSH–cytokine inverse coupling
    il6_z = (data["il6"] - np.nanmean(data["il6"])) / np.nanstd(data["il6"])
    tnf_z = (data["tnf_alpha"] - np.nanmean(data["tnf_alpha"])) / np.nanstd(data["tnf_alpha"])
    il8_z = (data["il8"] - np.nanmean(data["il8"])) / np.nanstd(data["il8"])
    cytokine_z = (il6_z + tnf_z + il8_z) / 3.0

    data["reduced_glutathione"] = np.clip(
        data["reduced_glutathione"] - cytokine_z * 15, 400, 1300
    ).round(1)

    # TG → LDL mild coupling
    tg_z = (data["triglycerides"] - np.nanmean(data["triglycerides"])) / np.nanstd(data["triglycerides"])
    data["ldl"] = np.clip(data["ldl"] + tg_z * 4, 50, 220).round(1)

    # Lipid → phospholipid coupling
    ldl_z = (data["ldl"] - np.nanmean(data["ldl"])) / np.nanstd(data["ldl"])
    lipid_z = (tg_z + ldl_z) / 2
    data["total_phospholipids"] = np.clip(
        data["total_phospholipids"] + lipid_z * 5, 100, 350
    ).round(1)

    # Refresh VLDL from TG
    data["vldl"] = np.clip(
        data["triglycerides"] / 5.0 + np.random.normal(0, 3.5, n), 5, 80
    ).round(1)

    # Re-clip everything to physiological ranges
    clips = {
        "serum_glutamate": (10, 120), "serine": (40, 200),
        "alanine": (100, 600), "lactate": (0.3, 3.5),
        "citrate": (40, 220), "ornithine": (15, 140),
        "citrulline": (8, 70), "arginine": (20, 160),
        "triglycerides": (40, 400), "ldl": (50, 220),
        "hdl": (20, 95), "vldl": (5, 80),
        "total_phospholipids": (100, 350), "sphingolipids": (100, 500),
        "reduced_glutathione": (400, 1300), "nitric_oxide": (8, 80),
        "il6": (0.3, 20), "tnf_alpha": (0.5, 22), "il8": (1.0, 30),
    }
    for col, (lo, hi) in clips.items():
        data[col] = np.clip(data[col], lo, hi)
        if col in ("lactate",):
            data[col] = np.round(data[col], 2)
        elif col in ("il6", "tnf_alpha", "il8"):
            data[col] = np.round(data[col], 2)
        else:
            data[col] = np.round(data[col], 1)

    return data


# ============================================================================
# 7. Introduce ~5 % missing data at random
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
# 8. Data Dictionary
# ============================================================================

DATA_DICTIONARY = pd.DataFrame([
    ("subject_id",   "int",          "-",       "Unique subject identifier"),
    ("diagnosis",    "int (binary)",  "-",       "Outcome: 1 = schizophrenia, 0 = healthy control"),
    ("age",          "int",          "years",   "Age at enrollment (18-70)"),
    ("sex",          "int (binary)",  "-",       "Biological sex: 1 = male, 0 = female"),
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
    ("serum_glutamate", "float", "µmol/L",  "Serum glutamate; primary excitatory neurotransmitter precursor"),
    ("serine",          "float", "µmol/L",  "Serum L-serine; NMDA receptor co-agonist precursor"),
    ("alanine",         "float", "µmol/L",  "Serum L-alanine; gluconeogenic amino acid"),
    ("lactate",         "float", "mmol/L",  "Serum lactate; glycolysis end-product / energy metabolism marker"),
    ("citrate",         "float", "µmol/L",  "Serum citrate; TCA cycle intermediate"),
    ("ornithine",   "float", "µmol/L",  "Serum ornithine; urea cycle intermediate"),
    ("citrulline",  "float", "µmol/L",  "Serum citrulline; urea cycle intermediate / NO synthesis marker"),
    ("arginine",    "float", "µmol/L",  "Serum L-arginine; NO synthase substrate / urea cycle"),
    ("triglycerides",       "float", "mg/dL",  "Serum triglycerides"),
    ("ldl",                 "float", "mg/dL",  "Low-density lipoprotein cholesterol"),
    ("hdl",                 "float", "mg/dL",  "High-density lipoprotein cholesterol"),
    ("vldl",                "float", "mg/dL",  "Very low-density lipoprotein cholesterol (≈ TG/5)"),
    ("total_phospholipids", "float", "mg/dL",  "Total serum phospholipids"),
    ("sphingolipids",       "float", "µmol/L", "Serum sphingolipids (ceramide pathway)"),
    ("reduced_glutathione", "float", "µmol/L", "Reduced glutathione (GSH); major intracellular antioxidant"),
    ("nitric_oxide",        "float", "µmol/L", "Serum nitric oxide metabolites (NOx); reflects NO bioavailability"),
    ("il6",       "float", "pg/mL", "Interleukin-6; pro-inflammatory cytokine"),
    ("tnf_alpha", "float", "pg/mL", "Tumor necrosis factor alpha; pro-inflammatory cytokine"),
    ("il8",       "float", "pg/mL", "Interleukin-8 (CXCL8); pro-inflammatory chemokine"),
], columns=["variable", "data_type", "unit", "description"])


# ============================================================================
# 9. Main Generation Pipeline
# ============================================================================

def main():
    print("Generating synthetic schizophrenia biomarker dataset ...")

    # 1. Demographics & diagnosis
    diagnosis, age, sex = generate_demographics(N_CASES, N_CONTROLS)

    # 2. Genetic variants
    gen_data = generate_genetic_variants(N_CASES, N_CONTROLS)

    # 3. Baseline biomarkers (SAME distribution for both groups)
    data = generate_baseline_biomarkers(N_TOTAL, age, sex)

    # 4. Apply weak linear shifts (d ≈ 0.08-0.15 per biomarker)
    data = apply_weak_main_effects(data, diagnosis)

    # 5. Apply STRONG nonlinear interactions (the real signal for ML)
    data = apply_nonlinear_interactions(data, diagnosis, age, sex, gen_data)

    # 6. Correlation tuning
    data = tune_correlations(data, N_TOTAL)

    # 7. Assemble DataFrame
    df = pd.DataFrame({
        "subject_id": np.arange(1, N_TOTAL + 1),
        "diagnosis": diagnosis,
        "age": age,
        "sex": sex,
    })

    for gene in VARIANT_CATALOG:
        df[f"{gene}_variant_present"] = gen_data[f"{gene}_variant_present"]
        df[f"{gene}_variant_id"] = gen_data[f"{gene}_variant_id"]

    biomarker_cols = [
        "serum_glutamate", "serine", "alanine", "lactate", "citrate",
        "ornithine", "citrulline", "arginine",
        "triglycerides", "ldl", "hdl", "vldl",
        "total_phospholipids", "sphingolipids",
        "reduced_glutathione", "nitric_oxide",
        "il6", "tnf_alpha", "il8",
    ]
    for col in biomarker_cols:
        df[col] = data[col]

    # 8. Introduce ~5 % missing data
    exclude = (
        ["subject_id", "diagnosis"]
        + [f"{g}_variant_present" for g in VARIANT_CATALOG]
        + [f"{g}_variant_id" for g in VARIANT_CATALOG]
    )
    df = introduce_missing(df, frac=0.05, exclude_cols=exclude)

    # 9. Save outputs
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

    print(f"\nInteraction subgroup sizes (cases only):")
    case = diagnosis == 1
    c4a = gen_data["C4A_variant_present"].astype(bool)
    comt_v = gen_data["COMT_variant_present"].astype(bool)
    drd2 = gen_data["DRD2_variant_present"].astype(bool)
    bdnf = gen_data["BDNF_variant_present"].astype(bool)
    grin2a = gen_data["GRIN2A_variant_present"].astype(bool)
    grm3 = gen_data["GRM3_variant_present"].astype(bool)
    disc1 = gen_data["DISC1_variant_present"].astype(bool)
    print(f"  C4A × COMT:           {(case & c4a & comt_v).sum()}")
    print(f"  DRD2 × BDNF:          {(case & drd2 & bdnf).sum()}")
    print(f"  GRIN2A × GRM3:        {(case & grin2a & grm3).sum()}")
    print(f"  Young × COMT:         {(case & (age < 30) & comt_v).sum()}")
    print(f"  DISC1 × BDNF:         {(case & disc1 & bdnf).sum()}")
    print(f"  C4A × DRD2 × male:    {(case & c4a & drd2 & (sex == 1)).sum()}")

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
