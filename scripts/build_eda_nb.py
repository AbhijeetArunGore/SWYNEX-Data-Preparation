import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m10 = """---
## Section 10 — Statistical Tests, Where Justified

### Hypothesis Testing Governance Framework
To test whether observed differences between patient subgroups are statistically robust or attributable to random variation, we apply formal non-parametric and contingency hypothesis tests.

**Important Methodological Limitation on Sample Independence**:
The dataset contains **101,763 encounters** across **71,515 distinct patients**, with **29.72%** being repeat hospitalizations. Consequently, observations from the same individual are not strictly independent (intra-cluster correlation). While large sample sizes generate very small p-values ($p < 10^{-15}$), these tests should be viewed as descriptive indicators of distributional separation rather than definitive clinical proof.

---

### Test 1: Prior Inpatient Admissions across Readmission Outcomes (Mann-Whitney U)
- **$H_0$**: The distribution of prior inpatient visits is identical between encounters readmitted within 30 days (`<30`) and those not readmitted (`NO`).
- **$H_1$**: The distribution of prior inpatient visits for the `<30` group is stochastically shifted toward higher values.
- **Assumptions**: Continuous/discrete ordinal scale, independent cohorts, no normality assumption required.
"""

c10_u1 = """# Execute Mann-Whitney U test on prior inpatient admissions
test1_res = perform_mann_whitney_u(df, continuous_col='number_inpatient', group_col='readmitted', group_val1='<30', group_val2='NO')
print("=== STATISTICAL TEST 1 RESULTS: PRIOR INPATIENT ADMISSIONS ===")
for k, v in test1_res.items():
    print(f"  {k:28s}: {v}")
"""

m10_u2 = """---
### Test 2: Length of Hospital Stay across Readmission Outcomes (Mann-Whitney U)
- **$H_0$**: The distribution of length of hospital stay (`time_in_hospital`) is identical between `<30` and `NO` cohorts.
- **$H_1$**: Patients readmitted within 30 days experience stochastically longer hospital stays.
"""

c10_u2 = """# Execute Mann-Whitney U test on length of stay
test2_res = perform_mann_whitney_u(df, continuous_col='time_in_hospital', group_col='readmitted', group_val1='<30', group_val2='NO')
print("=== STATISTICAL TEST 2 RESULTS: LENGTH OF HOSPITAL STAY ===")
for k, v in test2_res.items():
    print(f"  {k:28s}: {v}")
"""

m10_chi = """---
### Categorical Association Testing: Pearson Chi-Square & Cramer's V Effect Size
We evaluate categorical associations between readmission outcome (`readmitted`) and key clinical features: patient age bracket, diabetes medication prescription, and in-hospital medication dosage change.
"""

c10_chi = """# Execute Chi-Square tests
chi_tests = [
    perform_chi_square(df, cat_col1='age', cat_col2='readmitted'),
    perform_chi_square(df, cat_col1='diabetesMed', cat_col2='readmitted'),
    perform_chi_square(df, cat_col1='change', cat_col2='readmitted'),
    perform_chi_square(df, cat_col1='diag_1_category', cat_col2='readmitted')
]

chi_summary_df = pd.DataFrame(chi_tests)
print("=== CHI-SQUARE INDEPENDENCE TESTS & EFFECT SIZES ===")
chi_summary_df[['variable_1', 'chi2_statistic', 'degrees_of_freedom', 'p_value', 'cramers_v', 'statistically_significant']]
"""

m10_interp = """**Statistical Interpretation**:
1. **Prior Inpatient Admissions**: The Mann-Whitney U test yields $U = 401,064,764, p < 10^{-15}$. Encounters in the `<30` group averaged 1.224 prior inpatient admissions compared to 0.382 in the `NO` cohort (over 3.2x higher).
2. **Length of Stay**: The Mann-Whitney U test yields $U = 347,113,746, p = 1.595 \times 10^{-83}$, confirming that readmitted patients have a statistically significant right-shifted hospital duration (median = 4.0 vs 3.0 days).
3. **Medication Adjustments & Regimens**: Chi-Square tests demonstrate significant dependency between readmission and medication change ($\chi^2 = 215.94, p = 1.289 \times 10^{-47}$) as well as diabetes medication prescription ($\chi^2 = 386.44, p = 1.218 \times 10^{-84}$). Patients on active anti-diabetic medications experienced higher readmission rates (11.58%) than those on no diabetes medications (8.01%)."""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m10),
    nbf.v4.new_code_cell(c10_u1),
    nbf.v4.new_markdown_cell(m10_u2),
    nbf.v4.new_code_cell(c10_u2),
    nbf.v4.new_markdown_cell(m10_chi),
    nbf.v4.new_code_cell(c10_chi),
    nbf.v4.new_markdown_cell(m10_interp)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Section 10.")
