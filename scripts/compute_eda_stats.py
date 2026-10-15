import os
import pandas as pd
import numpy as np
from scipy import stats

csv_path = os.path.join('data', 'processed', 'diabetic_data_clean.csv')
df = pd.read_csv(csv_path)

print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Total encounters: {len(df):,}")
print(f"Unique patients: {df['patient_nbr'].nunique():,}")
repeat_rate = (1 - df['patient_nbr'].nunique() / len(df)) * 100
print(f"Repeat encounter percentage: {repeat_rate:.2f}%")

print("\n=== TARGET DISTRIBUTION (readmitted) ===")
counts = df['readmitted'].value_counts()
pcts = df['readmitted'].value_counts(normalize=True) * 100
for k in ['NO', '>30', '<30']:
    print(f"  {k:5s}: {counts[k]:,d} ({pcts[k]:.2f}%)")

df['readmit_30d'] = (df['readmitted'] == '<30').astype(int)
df['readmit_any'] = (df['readmitted'] != 'NO').astype(int)

print(f"Early readmission (<30): {df['readmit_30d'].sum():,} ({df['readmit_30d'].mean()*100:.2f}%)")
print(f"Any readmission (<30 or >30): {df['readmit_any'].sum():,} ({df['readmit_any'].mean()*100:.2f}%)")

print("\n=== NUMERICAL FEATURES DESCRIPTIVE STATS ===")
num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'age_midpoint']

stats_records = []
for c in num_cols:
    s = df[c]
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    stats_records.append({
        'Feature': c,
        'Mean': round(s.mean(), 2),
        'Std': round(s.std(), 2),
        'Median': round(s.median(), 2),
        'IQR': round(q75 - q25, 2),
        'Min': round(s.min(), 2),
        'Max': round(s.max(), 2),
        'Skew': round(s.skew(), 2)
    })
stats_df = pd.DataFrame(stats_records)
print(stats_df.to_string(index=False))

print("\n=== STATISTICAL TESTING ===")
# 1. Mann-Whitney U test: time_in_hospital between <30 and NO
los_30 = df[df['readmitted'] == '<30']['time_in_hospital']
los_no = df[df['readmitted'] == 'NO']['time_in_hospital']
u_stat, u_p = stats.mannwhitneyu(los_30, los_no, alternative='two-sided')
print(f"Mann-Whitney U: LOS (<30 vs NO): U={u_stat:,.0f}, p={u_p:.4e}, Median <30={los_30.median()}, Median NO={los_no.median()}")

# 2. Mann-Whitney U test: number_inpatient between <30 and NO
inp_30 = df[df['readmitted'] == '<30']['number_inpatient']
inp_no = df[df['readmitted'] == 'NO']['number_inpatient']
u_stat_inp, u_p_inp = stats.mannwhitneyu(inp_30, inp_no, alternative='two-sided')
print(f"Mann-Whitney U: Inpatient visits (<30 vs NO): U={u_stat_inp:,.0f}, p={u_p_inp:.4e}, Mean <30={inp_30.mean():.3f}, Mean NO={inp_no.mean():.3f}")

# 3. Chi-Square test: readmitted (3 classes) vs age bracket
contingency_age = pd.crosstab(df['age'], df['readmitted'])
chi2_age, p_age, dof_age, _ = stats.chi2_contingency(contingency_age)
print(f"Chi-square: Age vs Readmitted: Chi2={chi2_age:.2f}, p={p_age:.4e}, dof={dof_age}")

# 4. Chi-Square test: readmitted vs diabetesMed
contingency_med = pd.crosstab(df['diabetesMed'], df['readmitted'])
chi2_med, p_med, dof_med, _ = stats.chi2_contingency(contingency_med)
print(f"Chi-square: diabetesMed vs Readmitted: Chi2={chi2_med:.2f}, p={p_med:.4e}, dof={dof_med}")

# 5. Chi-Square test: readmitted vs change (medication change)
contingency_chg = pd.crosstab(df['change'], df['readmitted'])
chi2_chg, p_chg, dof_chg, _ = stats.chi2_contingency(contingency_chg)
print(f"Chi-square: Medication Change vs Readmitted: Chi2={chi2_chg:.2f}, p={p_chg:.4e}, dof={dof_chg}")

# 6. Correlation matrix
print("\n=== SPEARMAN CORRELATION MATRIX ===")
corr = df[num_cols].corr(method='spearman')
print(round(corr, 3).to_string())

print("\n=== BIVARIATE FINDINGS ===")
# Prior inpatient visits vs 30d readmission rate
print("Prior Inpatient Visits vs 30d Readmission Rate:")
inp_summary = df.groupby('number_inpatient').agg(
    total=('encounter_id', 'count'),
    readmit_30d_count=('readmit_30d', 'sum'),
    rate_30d=('readmit_30d', lambda x: round(x.mean() * 100, 2))
).reset_index()
print(inp_summary.head(7).to_string(index=False))

# Discharge Disposition vs 30d Readmission Rate (Top 8 categories)
print("\nDischarge Disposition vs 30d Readmission Rate:")
disch_summary = df.groupby('discharge_disposition_desc').agg(
    total=('encounter_id', 'count'),
    rate_30d=('readmit_30d', lambda x: round(x.mean() * 100, 2)),
    rate_any=('readmit_any', lambda x: round(x.mean() * 100, 2))
).sort_values(by='total', ascending=False).reset_index()
print(disch_summary.head(8).to_string(index=False))

# Primary Diagnosis Category vs 30d Readmission Rate
print("\nPrimary Diagnosis Category (diag_1_category) vs 30d Readmission Rate:")
diag_summary = df.groupby('diag_1_category').agg(
    total=('encounter_id', 'count'),
    rate_30d=('readmit_30d', lambda x: round(x.mean() * 100, 2)),
    rate_any=('readmit_any', lambda x: round(x.mean() * 100, 2))
).sort_values(by='total', ascending=False).reset_index()
print(diag_summary.to_string(index=False))

# A1C Result vs 30d Readmission Rate
print("\nA1Cresult vs 30d Readmission Rate:")
a1c_summary = df.groupby('A1Cresult').agg(
    total=('encounter_id', 'count'),
    rate_30d=('readmit_30d', lambda x: round(x.mean() * 100, 2)),
    rate_any=('readmit_any', lambda x: round(x.mean() * 100, 2))
).sort_values(by='total', ascending=False).reset_index()
print(a1c_summary.to_string(index=False))
