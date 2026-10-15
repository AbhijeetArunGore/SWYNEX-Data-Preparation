import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m4 = """---
## Section 4 — Data Dictionary and Variable Groups
To structure our exploratory analysis, we categorize the 54 features of the prepared dataset into seven functional clinical and administrative domains. This prevents misleading analytical treatments—such as calculating continuous averages for relational keys or treating record IDs as clinical predictors.

| Functional Domain | Key Variables Included | Data Types | Analytical Role & Modeling Treatment |
| :--- | :--- | :--- | :--- |
| **Identifiers** | `encounter_id`, `patient_nbr` | Integer / Object | Primary keys. `encounter_id` identifies the unit of observation; `patient_nbr` is the patient-level grouping key for leak-free cross-validation. Never used as predictive features. |
| **Demographics** | `age`, `age_midpoint`, `gender`, `race` | Categorical / Float | Baseline patient attributes. `age_midpoint` converts 10-year brackets into continuous midpoints. |
| **Healthcare Utilization** | `number_inpatient`, `number_emergency`, `number_outpatient` | Integer Counts | Counts of prior encounters in the preceding 12 months. Extreme right skew; captures historical acute acuity and healthcare dependency. |
| **Inpatient Complexity** | `time_in_hospital`, `num_medications`, `num_procedures`, `num_lab_procedures`, `number_diagnoses` | Integer Counts | Measures of encounter resource intensity and multi-morbidity burden during the index admission. |
| **Diagnoses** | `diag_1`, `diag_2`, `diag_3`, `diag_1_category`, `diag_2_category`, `diag_3_category` | Categorical | Primary, secondary, and tertiary diagnoses mapped to 9 high-level clinical categories from ICD-9 codes. |
| **Administrative & Transfer** | `admission_type_desc`, `discharge_disposition_desc`, `admission_source_desc`, `payer_code`, `medical_specialty` | Categorical | Operational characteristics of hospital admission and discharge pathways. Contains key risk stratifiers and terminal status leakage indicators (`Expired`). |
| **Pharmacotherapy & Glycemia** | `max_glu_serum`, `A1Cresult`, `change`, `diabetesMed`, 23 antidiabetic drug columns | Categorical | Glycemic control laboratory results (`Not Tested`, `Norm`, `>7`, `>8`), overall medication change status, and individual medication titration levels. |
| **Target Outcome** | `readmitted` | Categorical (`NO`, `>30`, `<30`) | Primary clinical outcome. `<30` indicates early readmission within the benchmark 30-day window; `>30` represents late readmission; `NO` indicates no recorded readmission. |
"""

c4 = """# Summary of feature distribution across functional domains
domain_mapping = {
    'Identifiers': ['encounter_id', 'patient_nbr'],
    'Demographics': ['age', 'age_midpoint', 'gender', 'race'],
    'Utilization': ['number_inpatient', 'number_emergency', 'number_outpatient'],
    'Inpatient Complexity': ['time_in_hospital', 'num_medications', 'num_procedures', 'num_lab_procedures', 'number_diagnoses'],
    'Diagnoses': ['diag_1', 'diag_2', 'diag_3', 'diag_1_category', 'diag_2_category', 'diag_3_category'],
    'Administrative & Transitions': ['admission_type_id', 'admission_type_desc', 'discharge_disposition_id', 'discharge_disposition_desc', 'admission_source_id', 'admission_source_desc', 'payer_code', 'medical_specialty'],
    'Pharmacotherapy & Glycemia': ['max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'] + [col for col in df.columns if col in ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']],
    'Target Outcome': ['readmitted']
}

domain_summary = pd.DataFrame([
    {'Domain': k, 'Feature Count': len(v), 'Sample Features': ', '.join(v[:3]) + ('...' if len(v) > 3 else '')}
    for k, v in domain_mapping.items()
])

print("=== CLINICAL DOMAIN BREAKDOWN (54 FEATURES) ===")
domain_summary
"""

m5 = """---
## Section 5 — Descriptive Statistics

### Numerical Feature Analysis
We calculate the complete suite of central tendency, dispersion, and distribution asymmetry metrics across all numerical variables.

**Clinical Insight on Mean vs. Median**:
For symmetric variables such as `num_lab_procedures` (Mean = 43.10, Median = 44.0, Skew = -0.24), the mean provides a representative measure of clinical activity. Conversely, for severely right-skewed utilization metrics such as `number_inpatient` (Mean = 0.64, Median = 0.0, Skew = +3.61) and `number_emergency` (Mean = 0.20, Median = 0.0, Skew = +22.86), the mean is heavily distorted by rare frequent-utilizer outliers. In these variables, the **median and Interquartile Range (IQR)** represent the only mathematically sound summary statistics.
"""

c5_num = """num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'age_midpoint']

num_summary = compute_numerical_summary(df, columns=num_cols)
print("=== COMPREHENSIVE NUMERICAL DESCRIPTIVE STATISTICS ===")
num_summary
"""

m5_cat = """### Categorical Feature Summary
We evaluate cardinalities, top frequency counts, and dominant categories across all demographic, administrative, and clinical features.
"""

c5_cat = """cat_cols = ['race', 'gender', 'age', 'admission_type_desc', 'discharge_disposition_desc',
            'admission_source_desc', 'diag_1_category', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'readmitted']

cat_summary = compute_categorical_summary(df, columns=cat_cols)
print("=== CATEGORICAL FEATURE ATTRIBUTE FREQUENCIES ===")
cat_summary
"""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m4),
    nbf.v4.new_code_cell(c4),
    nbf.v4.new_markdown_cell(m5),
    nbf.v4.new_code_cell(c5_num),
    nbf.v4.new_markdown_cell(m5_cat),
    nbf.v4.new_code_cell(c5_cat)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Sections 4 and 5.")
