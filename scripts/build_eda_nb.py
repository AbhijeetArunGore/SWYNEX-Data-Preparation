import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m12 = """---
## Section 12 — Evidence-Based Clinical Insights

Every insight in this section is strictly derived from the empirical calculations executed in this notebook and formatted under the standardized governance template: **Evidence → Interpretation → Potential Implication → Caveat**.

---

### Insight 1: Heavy Prior Inpatient Utilization Drives Exponential Escalation in 30-Day Readmission Risk
- **Evidence**: Encounters with 0 prior inpatient visits in the preceding 12 months had an **8.44%** 30-day readmission rate (N=67,627). This increased monotonically to **12.92%** at 1 visit (N=19,521), **17.43%** at 2 visits (N=7,566), **20.29%** at 3 visits (N=3,411), **23.61%** at 4 visits (N=1,622), and **34.58%** at 6 visits (N=480). Non-parametric testing confirmed statistical significance ($U = 401,064,764, p < 10^{-15}$).
- **Interpretation**: A high-frequency sub-cohort of chronically deteriorating patients accounts for an outsized proportion of hospital recidivism, reflecting persistent ambulatory care failure.
- **Potential Implication**: Prior utilization features must be engineered as primary risk stratifiers in Task 3 (e.g. non-linear splines, interaction ratios with emergency visits).
- **Caveat**: Utilization counts are observational and may capture socioeconomic barriers, insurance status, or primary care access rather than purely physiological disease severity.

---

### Insight 2: Substantial Target Imbalance Distorts Standard Classification Metrics
- **Evidence**: Early readmission within 30 days (`<30`) occurs in only **11.16%** of encounters (11,357 of 101,763), while non-readmissions comprise **53.91%** (54,861) and late readmissions (>30 days) comprise **34.93%** (35,545).
- **Interpretation**: The clinical benchmark exhibits an ~8:1 negative-to-positive imbalance. A naive zero-rule classifier predicting "No Readmission" across all encounters attains **88.84% accuracy** while offering zero clinical utility.
- **Potential Implication**: Model training in Task 3 must deploy cost-sensitive loss functions (`scale_pos_weight ≈ 7.95`), and model evaluation must be strictly governed by **PR-AUC**, **Brier score calibration**, and **Recall at fixed operational thresholds**.
- **Caveat**: Optimal operating thresholds cannot be determined in the abstract; they must be tuned against hospital economic costs and clinical intervention capacity.

---

### Insight 3: Post-Acute Facility Discharges Exhibit Markedly Higher Readmission Risk than Home Discharges
- **Evidence**: Patients discharged to rehabilitation facilities had a **27.71%** 30-day readmission rate (N=1,992), and transfers to Skilled Nursing Facilities (SNFs) had a **14.66%** rate (N=13,954), compared to **9.30%** for routine home discharges (N=60,232).
- **Interpretation**: Post-acute transitions represent vulnerable handoff points for elderly patients with profound functional impairment and multi-system medical complexity.
- **Potential Implication**: Discharge disposition should be used to trigger automated post-discharge transitional care management programs.
- **Caveat**: SNF transfers may experience higher readmission due to facility care protocol thresholds rather than intrinsic patient decompensation alone.

---

### Insight 4: In-Hospital Terminal Events Create Severe Target Leakage
- **Evidence**: 1,642 encounters resulted in inpatient patient death (`discharge_disposition_desc` = Expired), all exhibiting an exact **0.00%** readmission rate.
- **Interpretation**: Deceased patients are physically incapable of being readmitted. Including deceased encounters in predictive model training introduces catastrophic target leakage, allowing models to predict "no readmission" from end-of-life clinical markers.
- **Potential Implication**: All encounters with `discharge_disposition_id` in {11, 19, 20, 21} **must be removed** prior to training and evaluating Task 3 predictive models.
- **Caveat**: Excluding expired patients restricts the model scope to predicting readmission specifically among patients discharged alive.

---

### Insight 5: Length of Stay Moderately Correlates with Pharmacotherapy but Shows Limited Standalone Predictive Utility
- **Evidence**: Length of stay correlates moderately with the number of prescribed medications ($\rho = 0.465, p < 10^{-15}$), but median hospital duration differs by only 1 day between `<30` readmitted patients (4.0 days; mean = 4.77) and non-readmitted patients (3.0 days; mean = 4.18).
- **Interpretation**: Prolonged hospital stay captures inpatient clinical complexity and complication management, but operational discharge pressures bound the metric, making it insufficient as a standalone risk discriminator.
- **Potential Implication**: Feature engineering should combine length of stay into composite velocity ratios (`num_medications / time_in_hospital`) to capture intensity of care per day.
- **Caveat**: Length of stay is only known upon discharge. It cannot be utilized in admission-time triaging models without incurring temporal leakage.

---

### Insight 6: Systemic Under-Utilization of Inpatient Glycemic Testing (HbA1c)
- **Evidence**: 84,745 encounters (**83.28%**) had no documented HbA1c test ordered during admission (`Not Tested`), yet showed an **11.42%** 30-day readmission rate, compared to **9.66%** for normal tested patients and **9.87%** for high tested patients ($>8$).
- **Interpretation**: During 1999–2008, inpatient glycemic protocol compliance was inconsistent. Tested patients likely received structured endocrine attention, correlating with slightly lower readmission rates.
- **Potential Implication**: The presence of an HbA1c order serves as an informative proxy for institutional clinical protocol compliance.
- **Caveat**: Unmeasured confounders (e.g. non-diabetic primary admission reasons like orthopedics) strongly influenced whether glycemic testing was prioritized.

---

### Insight 7: Medication Regimen Titration ("Change") Signals Acute Glycemic Instability
- **Evidence**: Encounters where diabetes medication dosages were adjusted during hospitalization had a **12.24%** 30-day readmission rate (N=47,011), compared to **10.23%** for encounters with no regimen changes (N=54,752, $\chi^2 = 215.94, p = 1.289 \times 10^{-47}$).
- **Interpretation**: Medication changes flag poorly controlled blood glucose levels or new medication adjustments that require post-discharge stabilization, increasing acute vulnerability.
- **Potential Implication**: The `change` indicator represents an essential pre-discharge feature for risk scoring.
- **Caveat**: Medication changes may also capture vigilant clinical care; failing to adjust medications when indicated could lead to poorer long-term outcomes not captured in 30-day windows.

---

### Insight 8: High Intra-Patient Clustering Presents Critical Evaluation Leakage Risks
- **Evidence**: The 101,763 encounters stem from only 71,515 distinct patients. 30,248 encounters (**29.72%**) represent repeat admissions, with certain patients appearing up to 40 times.
- **Interpretation**: Encounters from the same patient are non-independent. Standard random splitting places encounters from the same patient into both training and test sets, artificially inflating model evaluation scores.
- **Potential Implication**: Model evaluation in Task 3 must enforce **GroupKFold** grouped by `patient_nbr` (or temporal prospective splitting) to simulate true out-of-sample patient prediction.
- **Caveat**: Patients with frequent historical records may benefit from longitudinal recurrent features, but models must be evaluated on unseen patients to establish genuine generalizability.
"""

nb.cells.append(nbf.v4.new_markdown_cell(m12))

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Section 12.")
