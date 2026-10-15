import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m13 = """---
## Section 13 — Implications for Future Modeling and Anti-Leakage Governance

The findings of this exploratory analysis establish critical rules and operational guardrails for downstream machine learning in Task 3:

### 1. Anti-Leakage Rule 1: Filtering In-Hospital Deceased Encounters
- **Empirical Finding**: 1,642 encounters resulted in in-hospital expiration (`discharge_disposition_id` in {11, 19, 20, 21}), exhibiting an exact 0.0% readmission rate.
- **Rule**: All expired encounters **must be excluded** from both training and evaluation cohorts prior to model development. Including them allows models to exploit end-of-life proxy variables to achieve false negative predictions.

### 2. Anti-Leakage Rule 2: Intra-Patient Grouped Evaluation Splits
- **Empirical Finding**: 29.72% of encounters represent repeat visits by the same individuals (71,515 distinct patients across 101,763 encounters).
- **Rule**: Standard random train/test splits violate sample independence and cause severe optimistic data leakage. Task 3 must enforce **GroupKFold cross-validation** grouped strictly on `patient_nbr`.

### 3. Feature Availability Timing: Admission vs. Discharge Scopes
- **Rule**: Variables must be strictly segregated based on operational inference timing:
  - **Admission-Time Triaging**: Uses only demographics, prior utilization (`number_inpatient`, `number_emergency`, `number_outpatient`), and admission source. Total length of stay, discharge destination, and full in-hospital medication counts are unavailable at admission and cannot be used without leakage.
  - **Pre-Discharge Transitional Care Planning**: Can incorporate total length of stay, cumulative procedures, discharge disposition, and in-hospital medication adjustments.

### 4. Evaluation Metrics for Severe Class Imbalance
- **Rule**: Overall accuracy must be rejected as an optimization objective. Model evaluation must be reported using **Precision-Recall AUC (PR-AUC)**, **Brier Score Calibration**, and **Recall / F1 at operational probability thresholds** tuned to clinical intervention capacity.
"""

m14 = """---
## Section 14 — Analytical Limitations

To maintain institutional integrity, the conclusions of this exploratory data analysis must be interpreted in light of the following constraints:

1. **Historical Observational Dataset (1999–2008)**: The data reflects clinical protocols, pharmacotherapy options, and documentation standards of over fifteen years ago. Modern clinical advances (e.g. SGLT2 inhibitors, GLP-1 receptor agonists, contemporary continuous glucose monitors) are absent.
2. **Observational Confounding**: All reported associations describe empirical correlations and do not imply causal relationships. Confounding by unmeasured variables (e.g. socioeconomic deprivation, health literacy, insurance benefit designs) cannot be eliminated.
3. **Administrative Coding Proxies**: Primary and secondary diagnoses are derived from ICD-9 billing codes, which are subject to hospital reimbursement incentives and coding variability.
4. **Encounter vs. Continuous Clinical State**: The dataset captures episodic inpatient encounters without outpatient electronic health record telemetry between admissions.
5. **No Direct Treatment Recommendations**: Results must not be construed as clinical guidance for individual patient medication selection or discharge timing.
"""

m15 = """---
## Section 15 — Executive Summary

### Analytical Synthesis of Task 2 Deliverables:
1. **Scope and Integrity**: Successfully profiled and validated **101,763 clean clinical inpatient encounters** across **71,515 unique patients** and **54 features** with zero unhandled missing values, strictly conforming to Task 1 data preparation baselines.
2. **Target Distribution**: Documented the baseline 3-class outcome (`NO`: 53.91%, `>30`: 34.93%, `<30`: 11.16%) and established the ~8:1 class imbalance governing 30-day early readmissions.
3. **Primary Predictive Drivers**:
   - Prior acute utilization (`number_inpatient`) is the most decisive risk indicator, escalating 30-day readmissions from 8.44% (0 visits) to 34.58% (6 visits; $p < 10^{-15}$).
   - Transitional discharge to rehabilitation facilities (27.71%) and SNFs (14.66%) indicates acute post-acute vulnerability.
   - In-hospital medication titration (`change`) flags acute glycemic instability (12.24% vs. 10.23% readmissions; $p < 10^{-15}$).
4. **Data Governance & Anti-Leakage Protocol**: Identified mandatory modeling pre-conditions:
   - Filter 1,642 in-hospital deceased records (`Expired`).
   - Enforce `GroupKFold(groups=patient_nbr)` cross-validation to isolate 29.72% repeat admissions.
   - Govern evaluation using PR-AUC and threshold-tuned recall.

---
**Task 2 Status**: Completed, fully validated, and ready for transition to **Task 3: Predictive Feature Engineering & Modeling**.
"""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m13),
    nbf.v4.new_markdown_cell(m14),
    nbf.v4.new_markdown_cell(m15)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Sections 13-15. Notebook assembly complete!")
