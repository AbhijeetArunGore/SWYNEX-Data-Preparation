import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m9 = """---
## Section 9 — Bivariate Analysis and Clinical Relationships

We evaluate relationships between patient features and readmission outcomes, reporting normalized subgroup rates, sample denominators, and clinical implications.
"""

c9_inp = """# 1. Readmission Risk vs. Prior Inpatient Utilization
inp_bivariate = compute_bivariate_rates(df, group_col='number_inpatient', target_col='readmitted')
print("=== PRIOR INPATIENT ADMISSIONS VS. READMISSION RATES ===")
inp_bivariate.head(8)
"""

c9_inp_vis = """# Plot prior inpatient visits vs. 30-day readmission rate
filtered_inp = inp_bivariate[inp_bivariate['Total_Encounters'] >= 100].copy()
filtered_inp['Inpatient_Visits'] = filtered_inp['Group_Value'].astype(int)
filtered_inp = filtered_inp.sort_values(by='Inpatient_Visits')

fig, ax1 = plt.subplots(figsize=(10, 5))
color_b = '#2b5c8f'
bars_i = ax1.bar(filtered_inp['Inpatient_Visits'], filtered_inp['Total_Encounters'], color=color_b, alpha=0.35, width=0.55, label='Encounter Count')
ax1.set_xlabel('Prior Inpatient Admissions in Preceding 12 Months', fontsize=11, fontweight='semibold')
ax1.set_ylabel('Total Encounters (Log Scale)', fontsize=11, fontweight='semibold', color=color_b)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor=color_b)

ax2 = ax1.twinx()
color_r = '#c0392b'
ax2.plot(filtered_inp['Inpatient_Visits'], filtered_inp['Readmit_30d_Rate_Pct'], color=color_r, marker='o', linewidth=2.5, markersize=7, label='30-Day Readmission Rate (%)')
ax2.set_ylabel('30-Day Readmission Rate (%)', fontsize=11, fontweight='semibold', color=color_r)
ax2.tick_params(axis='y', labelcolor=color_r)
ax2.set_ylim(0, 42)

for x, y in zip(filtered_inp['Inpatient_Visits'], filtered_inp['Readmit_30d_Rate_Pct']):
    ax2.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9.5, fontweight='bold', color=color_r)

plt.title('Prior Inpatient Utilization as Chief Readmission Driver (8.44% to 34.58%)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""

m9_inp_interp = """**Clinical Interpretation**: Prior inpatient hospitalization is the single most decisive predictor of readmission in this dataset. Patients with zero prior admissions exhibit an **8.44%** 30-day readmission rate. This escalates to **12.92%** at 1 visit, **20.29%** at 3 visits, and exceeds **34%** for chronic recidivists with 5+ visits. The step-change reflects severe underlying systemic illness and inadequate outpatient care stabilization."""

c9_disch = """# 2. Discharge Disposition vs. Readmission Risk
disch_bivariate = compute_bivariate_rates(df, group_col='discharge_disposition_desc', target_col='readmitted', min_count=500)
print("=== TOP DISCHARGE DESTINATIONS VS. READMISSION RATES ===")
disch_bivariate.head(10)
"""

c9_disch_vis = """# Visualize readmission by discharge disposition
plt.figure(figsize=(11, 5.5))
sorted_disch = disch_bivariate.sort_values(by='Readmit_30d_Rate_Pct', ascending=True)

# Truncate lengthy labels for visual clarity
short_labels = [s[:45] + ('...' if len(s) > 45 else '') for s in sorted_disch['Group_Value']]
bars_d = plt.barh(short_labels, sorted_disch['Readmit_30d_Rate_Pct'], color='#2b5c8f', edgecolor='black', linewidth=0.5)

plt.title('30-Day Readmission Rate by Primary Discharge Disposition Destination', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('30-Day Readmission Rate (%)', fontsize=11, fontweight='semibold')
plt.ylabel('Discharge Destination Category', fontsize=11, fontweight='semibold')
plt.xlim(0, 32)

for bar, n in zip(bars_d, sorted_disch['Total_Encounters']):
    w = bar.get_width()
    plt.text(w + 0.5, bar.get_y() + bar.get_height()/2., f'{w:.2f}% (N={n:,})', va='center', ha='left', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""

m9_disch_interp = """**Clinical Interpretation & Data Leakage Caveat**:
1. **Post-Acute Transitions**: Transfers to Rehabilitation Facilities (**27.71%**) and Skilled Nursing Facilities (**14.66%**) suffer substantially worse 30-day readmission rates than home discharges (**9.30%**), reflecting patient frailty and complex post-discharge care demands.
2. **Terminal Status Target Leakage**: 1,642 encounters resulted in in-hospital death (`Expired`), all showing an exact **0.00%** readmission rate. Patients who expire during their stay are physically incapable of post-discharge readmission. In Task 3, **all expired records must be excluded prior to training** to prevent the model from learning artificial shortcuts."""

c9_los_box = """# 3. Clinical Intensity: Length of Stay & Medications by Readmission Outcome
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Boxplot of Time in Hospital
sns.boxplot(data=df, x='readmitted', y='time_in_hospital', order=['NO', '>30', '<30'], palette=['#2b5c8f', '#e67e22', '#c0392b'], ax=axes[0])
axes[0].set_title('A: Length of Hospital Stay by Readmission Group', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Readmission Outcome', fontsize=10, fontweight='semibold')
axes[0].set_ylabel('Time in Hospital (Days)', fontsize=10, fontweight='semibold')

# Boxplot of Number of Medications
sns.boxplot(data=df, x='readmitted', y='num_medications', order=['NO', '>30', '<30'], palette=['#2b5c8f', '#e67e22', '#c0392b'], ax=axes[1])
axes[1].set_title('B: Medication Count by Readmission Group', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Readmission Outcome', fontsize=10, fontweight='semibold')
axes[1].set_ylabel('Number of Medications', fontsize=10, fontweight='semibold')

plt.suptitle('Comparison of Inpatient Complexity Metrics Across Readmission Cohorts', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""

m9_los_interp = """**Clinical Interpretation**: Patients readmitted within 30 days had longer median hospital stays (4.0 days; mean = 4.77) and higher median medication counts (15.0; mean = 16.71) compared to patients not readmitted (median stay = 3.0 days; mean = 4.18; mean medications = 15.67). While statistically significant, the overlap in interquartile ranges demonstrates that length of stay alone cannot serve as an isolated risk discriminator."""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m9),
    nbf.v4.new_code_cell(c9_inp),
    nbf.v4.new_code_cell(c9_inp_vis),
    nbf.v4.new_markdown_cell(m9_inp_interp),
    nbf.v4.new_code_cell(c9_disch),
    nbf.v4.new_code_cell(c9_disch_vis),
    nbf.v4.new_markdown_cell(m9_disch_interp),
    nbf.v4.new_code_cell(c9_los_box),
    nbf.v4.new_markdown_cell(m9_los_interp)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Section 9.")
