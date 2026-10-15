import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m8 = """---
## Section 8 — Target/Outcome Analysis

### Outcome Variable Definitions & Empirical Distribution
In the prepared dataset, the primary outcome is recorded in the `readmitted` feature, containing three distinct clinical categories:
- **`NO`**: No recorded readmission within the historical encounter capture window.
- **`>30`**: Readmission occurred more than 30 days following the index hospital discharge.
- **`<30`**: Readmission occurred within 30 days of hospital discharge.

### Binary Classification Target Formulations for Future Modeling
When framing readmission prediction as a binary classification problem for Task 3, two competing definitions exist in health informatics literature:
1. **Early Readmission Benchmark (CMS Protocol)**: `<30` (Positive Class, 1) vs. `NO` and `>30` (Negative Class, 0). This models the 30-day readmission window monitored by the Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP).
2. **Any Recorded Readmission**: `<30` and `>30` (Positive Class, 1) vs. `NO` (Negative Class, 0). This models general patient recidivism and chronic disease instability.

**Methodological Warning**: These definitions are fundamentally distinct and must never be treated as equivalent. Definition A isolates acute post-discharge breakdown, whereas Definition B captures generalized chronic disease burden.
"""

c8 = """# Compute exact outcome metrics
counts_3class = df['readmitted'].value_counts()[['NO', '>30', '<30']]
pcts_3class = (counts_3class / len(df)) * 100

df['readmit_30d'] = (df['readmitted'] == '<30').astype(int)
df['readmit_any'] = (df['readmitted'] != 'NO').astype(int)

target_summary = pd.DataFrame({
    'Readmission Category': counts_3class.index,
    'Encounter Count': counts_3class.values,
    'Cohort Percentage (%)': pcts_3class.values,
    'Clinical Definition': [
        'No readmission recorded in historical capture window',
        'Late readmission (>30 days post-discharge)',
        'Early readmission within CMS 30-day penalty threshold'
    ]
})

print("=== HOSPITAL READMISSION OUTCOME PREVALENCE ===")
target_summary
"""

m8_vis = """### Visualizing Class Distribution and Target Imbalance
We display the three-class outcome distribution alongside the binary 30-day early readmission formulation.
"""

c8_vis = """fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

# Left: 3-class distribution
bars1 = axes[0].bar(counts_3class.index, counts_3class.values, color=['#2b5c8f', '#e67e22', '#c0392b'], width=0.55, edgecolor='black', linewidth=0.5)
axes[0].set_title('A: Original 3-Class Readmission Distribution', fontsize=12, fontweight='bold', pad=12)
axes[0].set_xlabel('Readmission Category', fontsize=11, fontweight='semibold')
axes[0].set_ylabel('Hospital Encounters', fontsize=11, fontweight='semibold')
axes[0].set_ylim(0, 65000)

for bar, pct in zip(bars1, pcts_3class):
    y = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., y + 1200, f'{y:,}\\n({pct:.2f}%)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# Right: Binary 30-day distribution
bin_counts = pd.Series({
    'Not Readmitted <30 Days': len(df) - df['readmit_30d'].sum(),
    'Readmitted <30 Days (<30)': df['readmit_30d'].sum()
})
bin_pcts = (bin_counts / len(df)) * 100

bars2 = axes[1].bar(bin_counts.index, bin_counts.values, color=['#27ae60', '#c0392b'], width=0.45, edgecolor='black', linewidth=0.5)
axes[1].set_title('B: Binary CMS 30-Day Readmission Benchmark', fontsize=12, fontweight='bold', pad=12)
axes[1].set_xlabel('Classification Cohort', fontsize=11, fontweight='semibold')
axes[1].set_ylabel('Hospital Encounters', fontsize=11, fontweight='semibold')
axes[1].set_ylim(0, 105000)

for bar, pct in zip(bars2, bin_pcts):
    y = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., y + 1800, f'{y:,}\\n({pct:.2f}%)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.suptitle('Clinical Outcome Distribution & Severe Target Imbalance (N=101,763)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""

m8_interp = """**Analytical Finding on Class Imbalance**:
The primary target of interest (`<30`) occurs in only **11.16%** of encounters (11,357 cases). This reflects a **~8:1 negative-to-positive class imbalance**. Consequently:
1. **Accuracy is a Deceptive Metric**: A trivial null model that uniformly predicts "No 30-Day Readmission" yields an accuracy of **88.84%** while failing to identify a single vulnerable patient.
2. **Evaluation Strategy**: Model validation in Task 3 must prioritize **Precision-Recall AUC (PR-AUC)**, **Brier score calibration**, and **Recall at fixed decision thresholds** over raw accuracy or ROC-AUC alone."""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m8),
    nbf.v4.new_code_cell(c8),
    nbf.v4.new_markdown_cell(m8_vis),
    nbf.v4.new_code_cell(c8_vis),
    nbf.v4.new_markdown_cell(m8_interp)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Section 8.")
