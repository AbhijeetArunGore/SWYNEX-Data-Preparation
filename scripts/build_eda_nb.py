import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m6 = """---
## Section 6 — Univariate Analysis

We examine the marginal empirical distributions of key patient demographics, clinical complexity indices, and acute healthcare utilization metrics. Each visualization is paired with immediate analytical commentary and distribution metrics.
"""

c6_age = """# 1. Age Bracket Distribution
plt.figure(figsize=(10, 4.8))
age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
age_counts = df['age'].value_counts().reindex(age_order)
age_pcts = (age_counts / len(df)) * 100

bar_plot = sns.barplot(x=age_counts.index, y=age_counts.values, palette='Blues_d', edgecolor='black', linewidth=0.5)
plt.title('Univariate Distribution of Patient Age Brackets (N=101,763)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Age Bracket (Years)', fontsize=11, fontweight='semibold')
plt.ylabel('Hospital Encounters', fontsize=11, fontweight='semibold')
plt.ylim(0, 30000)

for p, pct in zip(bar_plot.patches, age_pcts):
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2., height + 600, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""

m6_age_interp = """**Interpretation**: The diabetic inpatient cohort is predominantly geriatric. Encounters among patients aged 60–90 constitute **64.6%** of all hospitalizations (70–80: 25.6%; 60–70: 22.1%; 80–90: 16.9%). Pediatric and adolescent diabetic admissions (<20 years) represent less than 0.9% of total encounters. Future predictive models must account for geriatric comorbidities and functional decline."""

c6_los = """# 2. Length of Stay (time_in_hospital)
fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={'height_ratios': [0.25, 0.75]}, sharex=True)

sns.boxplot(x=df['time_in_hospital'], color='#2b5c8f', ax=ax_box)
ax_box.set(xlabel='')
ax_box.set_title('Distribution of Length of Hospital Stay (Days) [Median=4.0, Mean=4.40, IQR=4.0]', fontsize=12, fontweight='bold')

sns.histplot(df['time_in_hospital'], discrete=True, color='#2b5c8f', kde=True, ax=ax_hist)
ax_hist.set_xlabel('Time in Hospital (Days)', fontsize=11, fontweight='semibold')
ax_hist.set_ylabel('Encounter Frequency', fontsize=11, fontweight='semibold')

plt.tight_layout()
plt.show()
"""

m6_los_interp = """**Interpretation**: Length of hospital stay exhibits a moderate right skew (+1.13) with a median of 4.0 days and an interquartile range of 4.0 days (25th percentile = 2.0 days; 75th percentile = 6.0 days). Over 53% of hospitalizations resolve within 3 days, but a prolonged inpatient tail extends to the dataset maximum of 14 days."""

c6_diag = """# 3. Primary Diagnosis Disease Categories (diag_1_category)
plt.figure(figsize=(10, 5))
diag_counts = df['diag_1_category'].value_counts()
diag_pcts = (diag_counts / len(df)) * 100

bars_d = sns.barplot(x=diag_counts.values, y=diag_counts.index, palette='crest_r', edgecolor='black', linewidth=0.5)
plt.title('Distribution of Primary Inpatient Diagnosis Categories (ICD-9)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Encounter Count', fontsize=11, fontweight='semibold')
plt.ylabel('Primary Clinical Category', fontsize=11, fontweight='semibold')
plt.xlim(0, 36000)

for p, pct in zip(bars_d.patches, diag_pcts):
    width = p.get_width()
    plt.text(width + 500, p.get_y() + p.get_height()/2., f'{width:,} ({pct:.1f}%)', va='center', ha='left', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""

m6_diag_interp = """**Interpretation**: Despite all patients having a recorded diagnosis of diabetes, the primary reason for acute hospital admission was **Circulatory Disease** in 29.9% of cases (30,436 encounters), followed by Respiratory Disease (14.2%) and Digestive conditions (9.3%). Primary acute diabetes admissions accounted for 8.6% of encounters (8,757), demonstrating that cardiovascular manifestations dominate diabetic inpatient morbidity."""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m6),
    nbf.v4.new_code_cell(c6_age),
    nbf.v4.new_markdown_cell(m6_age_interp),
    nbf.v4.new_code_cell(c6_los),
    nbf.v4.new_markdown_cell(m6_los_interp),
    nbf.v4.new_code_cell(c6_diag),
    nbf.v4.new_markdown_cell(m6_diag_interp)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Section 6.")
