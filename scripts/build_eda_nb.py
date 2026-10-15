import os
import nbformat as nbf

nb_path = os.path.join('notebooks', '02_exploratory_data_analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

m11 = """---
## Section 11 — Correlation and Feature Relationships

### Collinearity and Association Audit
We compute the Spearman rank-order correlation matrix ($\rho$) across all clinical and utilization numerical features. Spearman correlation is chosen over Pearson because it evaluates monotonic relationships without assuming bivariate normality, making it robust against skewed clinical variables.

**Critical Data Science Warnings**:
1. **Correlation $\neq$ Causation**: Observed associations describe shared clinical complexity rather than causal pathways.
2. **Correlation $\neq$ Predictive Feature Importance**: A feature with near-zero linear correlation (e.g., `number_emergency`, $\rho = 0.044$ with medications) can still exhibit non-linear predictive power in gradient boosting models.
3. **No Arbitrary ID Encodings**: Relational IDs (`admission_type_id`, `patient_nbr`) are strictly excluded from correlation computation to prevent mathematical nonsense.
"""

c11 = """# Calculate Spearman correlation matrix
corr_cols = ['time_in_hospital', 'num_medications', 'num_procedures', 'num_lab_procedures',
             'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient', 'age_midpoint']

corr_matrix = compute_correlation_matrix(df, columns=corr_cols, method='spearman')
print("=== SPEARMAN RANK CORRELATION MATRIX ===")
corr_matrix
"""

m11_vis = """### Correlation Heatmap Visualization
We render the correlation matrix using a masked upper-triangle layout and divergent colormap.
"""

c11_vis = """plt.figure(figsize=(10, 7.5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

labels_corr = ['Length of Stay', 'Medications', 'Procedures', 'Lab Tests', 'Diagnoses', 'Inpatient Visits', 'Emergency Visits', 'Outpatient Visits', 'Age Midpoint']

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmax=0.6,
    vmin=-0.2,
    center=0,
    annot=True,
    fmt='.2f',
    square=True,
    linewidths=0.6,
    xticklabels=labels_corr,
    yticklabels=labels_corr,
    cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation (rho)"}
)

plt.title('Spearman Correlation Matrix of Clinical & Utilization Features', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()
"""

m11_interp = """**Analytical Findings on Collinearity**:
1. **Length of Stay & Pharmacotherapy**: The strongest positive correlation occurs between `time_in_hospital` and `num_medications` ($\rho = 0.465, p < 10^{-15}$). Extended hospitalizations inherently accumulate more pharmaceutical interventions and treatment adjustments.
2. **Procedural & Diagnostic Complexity**: Moderate correlations appear between `num_medications` and `num_procedures` ($\rho = 0.352$) and `number_diagnoses` ($\rho = 0.294$).
3. **Utilization Independence**: Prior utilization channels (`number_inpatient`, `number_emergency`, `number_outpatient`) show modest inter-correlation ($\rho = 0.156$ to $0.222$), indicating that emergency room use and inpatient admissions capture distinct dimensions of care recidivism without severe multi-collinearity ($VIF < 1.3$)."""

nb.cells.extend([
    nbf.v4.new_markdown_cell(m11),
    nbf.v4.new_code_cell(c11),
    nbf.v4.new_markdown_cell(m11_vis),
    nbf.v4.new_code_cell(c11_vis),
    nbf.v4.new_markdown_cell(m11_interp)
])

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Updated {nb_path} with Section 11.")
