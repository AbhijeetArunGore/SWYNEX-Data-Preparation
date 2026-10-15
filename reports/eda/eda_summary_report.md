# SWYNEX Technologies — Task 2: Exploratory Data Analysis (EDA)
## Executive Clinical Data Science & Findings Report

**Project**: Real-World Diabetes Inpatient Encounters Analysis (1999–2008)  
**Author**: Abhijeet Arun Gore  
**Role**: Data Science Intern, SWYNEX Technologies  
**Baseline Dataset**: Cleaned Inpatient Dataset from Task 1 (`data/processed/diabetic_data_clean.csv`)  
**Scope**: 101,763 hospital encounters across 130 US hospitals, 54 features, 71,515 unique patients  

---

## 1. Executive Summary
This report delivers the analytical findings from **Task 2: Exploratory Data Analysis (EDA)** conducted on the prepared Diabetes 130-US Hospitals dataset. The objective of Task 2 is to uncover statistical distributions, identify non-linear relationships, conduct hypothesis testing, evaluate class imbalance, audit clinical collinearity, and extract evidence-based operational insights to govern downstream predictive modeling for 30-day hospital readmissions.

### Key Headline Metrics:
- **Analyzed Cohort**: 101,763 inpatient encounters across 71,515 unique patients (29.72% repeat admission rate).
- **Target Outcome Prevalence**:
  - `NO` (No recorded readmission): 54,861 encounters (53.91%)
  - `>30` (Readmitted after 30 days): 35,545 encounters (34.93%)
  - `<30` (Readmitted within 30 days — primary benchmark): 11,357 encounters (11.16%)
- **Primary Readmission Driver**: Prior inpatient utilization in the preceding 12 months demonstrated the steepest risk gradient, escalating 30-day readmission from **8.44%** (0 visits) to **34.58%** (6 visits).
- **Post-Acute Transfer Vulnerability**: Discharges to rehabilitation facilities (27.71%) and Skilled Nursing Facilities (14.66%) carried substantially higher 30-day readmission risk compared to home discharges (9.30%).
- **Critical Leakage Alert**: 1,642 deceased patients (`Expired`) were identified in the dataset with 0.0% readmission rate. These records represent post-discharge terminal events and **must be excluded** prior to training predictive models.

---

## 2. Dataset Profiling & Analytical Schema

| Dimension | Value / Specification | Analytical Interpretation |
| :--- | :--- | :--- |
| **Total Encounters** | 101,763 | Unit of observation is an encounter, not an individual patient. |
| **Distinct Patients** | 71,515 | 30,248 admissions are repeat hospitalizations by recurring patients. |
| **Repeat Admission Rate** | 29.72% | Demands **GroupKFold** patient-level splitting to avoid data leakage. |
| **Total Features** | 54 clean features | Comprising demographics, utilization counts, medications, diagnoses, and lab tests. |
| **Unhandled Nulls** | 0 (0.00%) | Maintained from Task 1 preprocessing standards. |
| **Memory Footprint** | ~56.2 MB in memory | Compact and reproducible for high-throughput exploratory computing. |

---

## 3. Univariate Distributions & Skewness Analysis

| Feature Name | Mean | Std | Median | IQR | Min | Max | Skewness | Distribution Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `time_in_hospital` | 4.40 | 2.99 | 4.0 | 4.0 | 1.0 | 14.0 | +1.13 | Moderate right skew; median of 4 days is more robust than mean. |
| `num_medications` | 16.02 | 8.13 | 15.0 | 10.0 | 1.0 | 81.0 | +1.33 | Right-skewed polypharmacy indicator; long tail up to 81 medications. |
| `num_lab_procedures` | 43.10 | 19.67 | 44.0 | 26.0 | 1.0 | 132.0 | -0.24 | Symmetrical, bell-shaped distribution across hospital stays. |
| `num_procedures` | 1.34 | 1.71 | 1.0 | 2.0 | 0.0 | 6.0 | +1.32 | Discrete count; 46.1% of encounters had zero non-lab procedures. |
| `number_inpatient` | 0.64 | 1.26 | 0.0 | 1.0 | 0.0 | 21.0 | +3.61 | Extreme right skew; 66.5% zero visits, long tail of chronic utilizers. |
| `number_emergency` | 0.20 | 0.93 | 0.0 | 0.0 | 0.0 | 76.0 | +22.86 | Highly sparse hurdle variable; 88.9% zero emergency visits. |
| `number_outpatient` | 0.37 | 1.27 | 0.0 | 0.0 | 0.0 | 42.0 | +8.83 | Sparse ambulatory utilization metric. |
| `number_diagnoses` | 7.42 | 1.93 | 8.0 | 3.0 | 1.0 | 16.0 | -0.88 | Left-skewed; modal cap at 9 diagnoses reflecting CMS billing caps. |
| `age_midpoint` | 65.97 | 15.94 | 65.0 | 20.0 | 5.0 | 95.0 | -0.63 | Geriatric concentration; >67% of encounters involve patients aged 60+. |

---

## 4. Key Bivariate Relationships & Clinical Findings

### A. Prior Inpatient Utilization Gradient
Historical hospital admissions within the preceding 12 months represent the single most potent empirical indicator of 30-day readmission:

| Prior Inpatient Visits | Total Encounters | Cohort % | 30-Day Readmissions | 30-Day Readmission Rate (%) | Relative Risk vs 0 Visits |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 67,627 | 66.46% | 5,706 | **8.44%** | 1.00x (Baseline) |
| **1** | 19,521 | 19.18% | 2,523 | **12.92%** | 1.53x |
| **2** | 7,566 | 7.43% | 1,319 | **17.43%** | 2.07x |
| **3** | 3,411 | 3.35% | 692 | **20.29%** | 2.40x |
| **4** | 1,622 | 1.59% | 383 | **23.61%** | 2.80x |
| **5** | 812 | 0.80% | 255 | **31.40%** | 3.72x |
| **6+** | 1,204 | 1.18% | 479 | **39.78%** | 4.71x |

### B. Discharge Destination & Post-Acute Risk

| Discharge Disposition Category | Total Encounters | 30-Day Readmit Rate (%) | Any Readmission Rate (%) | Clinical Risk Profile |
| :--- | :---: | :---: | :---: | :--- |
| **Discharged to Home** | 60,232 | **9.30%** | 45.02% | Lowest acute risk baseline. |
| **Home with Home Health Service** | 12,902 | **12.70%** | 54.26% | Moderate vulnerability; frequent follow-up needs. |
| **Skilled Nursing Facility (SNF)** | 13,954 | **14.66%** | 49.89% | Elevated frailty, chronic functional dependency. |
| **Another Short Term Hospital** | 2,128 | **16.07%** | 47.46% | Inter-facility transfer indicating clinical instability. |
| **Another Rehab Facility** | 1,992 | **27.71%** | 53.77% | Highest acute risk group; high therapeutic complexity. |
| **Expired (Deceased)** | 1,642 | **0.00%** | 0.00% | Terminal outcome; **leakage source** for future modeling. |

---

## 5. Inferential Statistical Hypothesis Testing

### Test 1: Prior Inpatient Admissions across Readmission Outcomes (Mann-Whitney U)
- **Null Hypothesis ($H_0$)**: The distribution of prior inpatient visits is identical between encounters readmitted within 30 days (`<30`) and those not readmitted (`NO`).
- **Alternative Hypothesis ($H_1$)**: Encounters readmitted within 30 days have a stochastically greater number of prior inpatient visits.
- **Statistical Result**: Mann-Whitney $U = 401,064,764$, $p < 10^{-15}$ (Two-sided).
- **Empirical Metrics**: Mean for `<30` group = **1.224 visits** vs Mean for `NO` group = **0.382 visits** (over 3.2x higher).
- **Conclusion**: Reject $H_0$. Prior acute care utilization exhibits an extraordinarily strong, statistically significant association with early readmission.

### Test 2: Length of Hospital Stay between Readmitted and Non-Readmitted Patients (Mann-Whitney U)
- **Null Hypothesis ($H_0$)**: Length of stay distributions are identical between `<30` and `NO` groups.
- **Statistical Result**: Mann-Whitney $U = 347,113,746$, $p = 1.595 \times 10^{-83}$.
- **Empirical Metrics**: Median for `<30` = **4.0 days** (Mean = 4.77) vs Median for `NO` = **3.0 days** (Mean = 4.18).
- **Conclusion**: Reject $H_0$. Extended hospital stays reflect greater underlying acute severity or in-hospital complications.

### Test 3: Association between Age Bracket and Readmission Outcome (Pearson Chi-Square)
- **Null Hypothesis ($H_0$)**: Patient age bracket is statistically independent of readmission outcome.
- **Statistical Result**: $\chi^2 = 313.33$, $\text{df} = 18$, $p = 8.676 \times 10^{-56}$, Cramer's $V = 0.0392$.
- **Conclusion**: Statistically significant dependency driven by elevated readmission rates among patients aged 70–80 (11.83%) and 80–90 (11.87%) compared to younger cohorts (<30: ~8.8%).

### Test 4: Association between Medication Change Status and Readmission (Pearson Chi-Square)
- **Null Hypothesis ($H_0$)**: Diabetes medication titration during admission is independent of readmission outcome.
- **Statistical Result**: $\chi^2 = 215.94$, $\text{df} = 2$, $p = 1.289 \times 10^{-47}$, Cramer's $V = 0.0461$.
- **Conclusion**: Encounters with medication adjustments had higher 30-day readmission (**12.24%**) compared to encounters with stable medication regimens (**10.23%**).

---

## 6. Eight Evidence-Based Clinical Insights

### Insight 1: Heavy Inpatient Utilization is the Dominant Leading Indicator of 30-Day Readmission
- **Evidence**: Patients with zero prior inpatient visits experienced an **8.44%** 30-day readmission rate (N=67,627), whereas patients with 1 visit experienced **12.92%**, 3 visits reached **20.29%**, and 6+ visits reached **39.78%** ($U = 401,064,764, p < 10^{-15}$).
- **Interpretation**: A small subgroup of "frequent flyers" drives an outsized proportion of emergency and inpatient readmissions, reflecting unmanaged multi-morbidity or post-discharge vulnerability.
- **Potential Implication**: Feature engineering should construct cumulative utilization velocity ratios (`inpatient_visits / (outpatient_visits + 1)`) as top candidate predictors.
- **Caveat**: Utilization counts are historical observations and may correlate with socioeconomic barriers, access to primary care, or insurance status rather than purely biological disease severity.

### Insight 2: Substantial Target Imbalance Distorts Standard Classification Accuracy
- **Evidence**: The primary operational target—readmission within 30 days (`<30`)—comprises only **11.16%** of encounters (11,357 of 101,763), while non-readmissions comprise **53.91%** and readmissions beyond 30 days comprise **34.93%**.
- **Interpretation**: A naive majority-class classifier predicting "No 30-Day Readmission" would achieve **88.84% accuracy** while missing 100% of high-risk patients.
- **Potential Implication**: Future modeling must abandon overall accuracy in favor of threshold-independent **PR-AUC**, **ROC-AUC**, and high-recall operating points. Cost-sensitive learning or class weighting must be applied.
- **Caveat**: The appropriate threshold depends on hospital economics and intervention costs (e.g. nurse home visits vs CMS penalty avoidance).

### Insight 3: Post-Acute Facility Discharges Suffer Elevated Re-Hospitalization Rates
- **Evidence**: Patients discharged to rehabilitation units experienced a **27.71%** 30-day readmission rate, and transfers to Skilled Nursing Facilities (SNFs) had a **14.66%** rate, compared to **9.30%** for home discharges.
- **Interpretation**: Patients requiring transitional sub-acute care possess higher clinical frailty and impaired functional status, making transition handoffs a critical point of failure.
- **Potential Implication**: Discharge disposition should be treated as a primary stratifier for post-discharge intervention programs.
- **Caveat**: SNF transfers may experience higher readmission due to facility care protocol thresholds rather than intrinsic patient decompensation alone.

### Insight 4: Terminal Discharge Events Introduce Lethal Target Leakage
- **Evidence**: 1,642 encounters resulted in patient death during hospitalization (`discharge_disposition_id` 11, 19, 20, 21 - Expired), all exhibiting an exact **0.00%** readmission rate.
- **Interpretation**: Deceased patients are physically incapable of being readmitted. Including them in predictive model training creates artificial target leakage where models learn to predict "no readmission" from terminal end-of-life indicators.
- **Potential Implication**: Prior to training any Task 3 predictive models, all expired encounters **must be strictly filtered out** from both train and test cohorts.
- **Caveat**: Excluding expired patients changes the model scope strictly to "patients discharged alive."

### Insight 5: Length of Stay Moderately Correlates with Clinical Complexity but Shows Limited Standalone Predictive Power
- **Evidence**: Length of stay correlates moderately with the number of prescribed medications ($\rho = 0.465, p < 10^{-15}$) and number of diagnoses ($\rho = 0.237$), but median stay differs by only 1 day between readmitted (4.0 days) and non-readmitted patients (3.0 days).
- **Interpretation**: Hospital duration reflects inpatient complexity and clinical stability but is bounded by hospital operational discharge pressures, making it insufficient as a standalone risk predictor.
- **Potential Implication**: Combine length of stay with medication velocity (`medications / time_in_hospital`) to capture intensity of care per day.
- **Caveat**: Length of stay is only known upon discharge; models intended for admission-time triaging cannot utilize total length of stay without leakage.

### Insight 6: Extensive Omission of Glycemic Testing (HbA1c) in Inpatient Encounters
- **Evidence**: 84,745 encounters (**83.28%**) had no documented HbA1c lab test ordered (`Not Tested`), yet showed an **11.42%** 30-day readmission rate compared to **9.66%** for normal tested patients and **9.87%** for elevated tested patients ($>8$).
- **Interpretation**: During the 1999–2008 timeframe, inpatient glycemic protocol compliance was inconsistent. Tested patients likely received structured endocrine attention, correlating with slightly lower readmission rates.
- **Potential Implication**: The presence of an HbA1c test order serves as an informative proxy for clinical protocol compliance and physician attentiveness rather than purely biological risk.
- **Caveat**: Unmeasured confounders (e.g., patient admission under non-diabetic services) strongly influence whether glucose testing was prioritized.

### Insight 7: Medication TITRATION ("Change") Flags Acute Unstable Glycemia
- **Evidence**: Patients whose diabetes medications were changed during hospitalization had a **12.24%** 30-day readmission rate (N=47,011), compared to **10.23%** for patients with no regimen changes (N=54,752, $\chi^2 = 215.94, p = 1.289 \times 10^{-47}$).
- **Interpretation**: Medication adjustments indicate unmanaged blood glucose levels or therapy transitions that require post-discharge stabilization, increasing acute vulnerability.
- **Potential Implication**: The `change` indicator represents an essential pre-discharge feature for risk scoring.
- **Caveat**: Medication changes may also capture vigilant clinical care; failing to adjust medications when indicated could lead to poorer long-term outcomes not captured in 30-day windows.

### Insight 8: High Intra-Patient Clustering Presents Critical Evaluation Leakage Risks
- **Evidence**: The 101,763 encounters stem from only 71,515 distinct patients. 30,248 encounters (**29.72%**) represent repeat admissions, with certain patients appearing up to 40 times.
- **Interpretation**: Encounters from the same patient are non-independent. Standard random splitting places encounters from the same patient into both training and test sets, artificially inflating model evaluation scores.
- **Potential Implication**: Model evaluation in Task 3 must enforce **GroupKFold** grouped by `patient_nbr` (or temporal prospective splitting) to simulate true out-of-sample patient prediction.
- **Caveat**: Patients with frequent historical records may benefit from longitudinal recurrent features, but models must be evaluated on unseen patients to establish genuine generalizability.

---

## 7. Modeling Implications & Anti-Leakage Governance Matrix

| Risk / Consideration | Empirical Evidence | Proposed Engineering & Validation Rule |
| :--- | :--- | :--- |
| **Severe Class Imbalance** | 11.16% positive rate (`<30`) | Utilize **PR-AUC** and **F1-score** at tuned operating thresholds; apply focal loss or class weights (`scale_pos_weight ≈ 8.0`). |
| **Terminal Status Leakage** | 1,642 Expired patients (0.0% readmit) | **Mandatory Filtering**: Remove `discharge_disposition_id` $\in \{11, 19, 20, 21\}$ before model training. |
| **Intra-Patient Clustering** | 29.72% repeat encounters | **Grouped Cross-Validation**: Enforce `GroupKFold(groups=df['patient_nbr'])`. |
| **Post-Discharge Feature Leakage** | Discharge disposition timing | Features like `discharge_disposition` are valid only for discharge-time scoring, invalid for admission-time triage. |
| **Non-Linear Utilization Impact** | Inpatient visits jump from 8.4% to 34.6% | Apply non-linear models (LightGBM/XGBoost) or splines/binning for linear models. |
| **Right-Skewed Continuous Scales** | Emergency skew = 22.86, Inpatient = 3.61 | Log-transform ($\log(1+x)$) or robust quantile scaling for parametric models. |

---

## 8. Limitations & Methodological Caveats
1. **Historical Observational Nature**: Data originates from 1999–2008 and reflects historical clinical standards, coding practices, and pharmacological formulations.
2. **Observational Confounding**: Associations identified in this EDA cannot establish causal relationships.
3. **Hospital Coverage & Generalizability**: The dataset covers 130 hospitals primarily located in the United States; results may not generalize across international health systems.
4. **Encounter vs Patient Granularity**: Encounter-level records aggregate multi-morbid hospital stays without continuous outpatient EHR monitoring.
5. **Billing Code Proxies**: ICD-9 diagnoses reflect administrative billing codes that may prioritize reimbursement thresholds over nuanced pathophysiology.

---

## 9. Next Steps for Task 3
- [ ] Implement strict filtering of 1,642 deceased encounters.
- [ ] Engineer interaction features (`utilization_velocity`, `los_medication_ratio`, `icd9_charlson_comorbidity_proxy`).
- [ ] Construct a leak-free `GroupKFold` validation harness.
- [ ] Benchmark baseline classifiers (Logistic Regression, Random Forest, LightGBM) optimized on PR-AUC.
