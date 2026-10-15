"""SWYNEX EDA Figure Generation Script.

Generates and exports publication-quality visualizations for the SWYNEX
Exploratory Data Analysis report and presentation materials.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure professional styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

DATA_PATH = os.path.join('data', 'processed', 'diabetic_data_clean.csv')
OUTPUT_DIR = os.path.join('reports', 'eda', 'figures')


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['readmit_30d'] = (df['readmitted'] == '<30').astype(int)
    df['readmit_any'] = (df['readmitted'] != 'NO').astype(int)
    return df


def generate_target_distribution_figure(df: pd.DataFrame, output_dir: str):
    """Figure 1: Target Outcome Class Distribution & 30-Day Rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: 3-Class Categorical Target
    target_counts = df['readmitted'].value_counts()[['NO', '>30', '<30']]
    target_pcts = (target_counts / len(df)) * 100
    colors_3class = ['#2b5c8f', '#e67e22', '#c0392b']

    bars = axes[0].bar(target_counts.index, target_counts.values, color=colors_3class, width=0.55, edgecolor='black', linewidth=0.5)
    axes[0].set_title('A: Original Hospital Readmission Outcome Distribution', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_xlabel('Readmission Category', fontsize=11, fontweight='semibold')
    axes[0].set_ylabel('Hospital Encounters', fontsize=11, fontweight='semibold')
    axes[0].set_ylim(0, 65000)

    for bar, pct in zip(bars, target_pcts):
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, yval + 1200, f'{yval:,}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel B: Binary 30-Day Early Readmission
    bin_counts = pd.Series({
        'Not Readmitted <30 Days': len(df) - df['readmit_30d'].sum(),
        'Readmitted <30 Days': df['readmit_30d'].sum()
    })
    bin_pcts = (bin_counts / len(df)) * 100
    colors_bin = ['#27ae60', '#c0392b']

    bars_b = axes[1].bar(bin_counts.index, bin_counts.values, color=colors_bin, width=0.45, edgecolor='black', linewidth=0.5)
    axes[1].set_title('B: Binary Classification Cohort (30-Day Readmission Benchmark)', fontsize=12, fontweight='bold', pad=12)
    axes[1].set_xlabel('Clinical Classification Cohort', fontsize=11, fontweight='semibold')
    axes[1].set_ylabel('Hospital Encounters', fontsize=11, fontweight='semibold')
    axes[1].set_ylim(0, 105000)

    for bar, pct in zip(bars_b, bin_pcts):
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 1800, f'{yval:,}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('SWYNEX Technologies — Clinical Readmission Outcome Profiling (N=101,763)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, '01_target_distribution.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_numerical_distributions_figure(df: pd.DataFrame, output_dir: str):
    """Figure 2: Distribution of Key Clinical Numerical Variables."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Length of stay
    sns.histplot(df['time_in_hospital'], bins=14, discrete=True, kde=True, color='#2b5c8f', ax=axes[0, 0])
    axes[0, 0].set_title('A: Length of Hospital Stay (Days) [Median=4, Skew=1.13]', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Time in Hospital (Days)', fontsize=10)
    axes[0, 0].set_ylabel('Encounter Frequency', fontsize=10)

    # 2. Number of medications
    sns.histplot(df['num_medications'], bins=30, kde=True, color='#27ae60', ax=axes[0, 1])
    axes[0, 1].set_title('B: Number of Prescribed Medications [Median=15, Skew=1.33]', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Medications', fontsize=10)
    axes[0, 1].set_ylabel('Encounter Frequency', fontsize=10)

    # 3. Lab procedures
    sns.histplot(df['num_lab_procedures'], bins=35, kde=True, color='#e67e22', ax=axes[1, 0])
    axes[1, 0].set_title('C: Number of Lab Procedures [Median=44, Skew=-0.24]', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Lab Procedures', fontsize=10)
    axes[1, 0].set_ylabel('Encounter Frequency', fontsize=10)

    # 4. Number of diagnoses
    sns.histplot(df['number_diagnoses'], bins=16, discrete=True, kde=False, color='#8e44ad', ax=axes[1, 1])
    axes[1, 1].set_title('D: Number of Coded Diagnoses [Median=8, Skew=-0.88]', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Diagnoses', fontsize=10)
    axes[1, 1].set_ylabel('Encounter Frequency', fontsize=10)

    plt.suptitle('SWYNEX Technologies — Univariate Distribution of Clinical Complexity Metrics', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = os.path.join(output_dir, '02_numerical_distributions.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_prior_inpatient_figure(df: pd.DataFrame, output_dir: str):
    """Figure 3: Prior Inpatient Visits vs 30-Day Readmission Risk."""
    summary = df.groupby('number_inpatient').agg(
        total=('encounter_id', 'count'),
        readmit_30d=('readmit_30d', 'sum')
    ).reset_index()
    summary['rate_30d'] = (summary['readmit_30d'] / summary['total']) * 100
    filtered = summary[summary['number_inpatient'] <= 6].copy()

    fig, ax1 = plt.subplots(figsize=(11, 6))

    color_bar = '#34495e'
    bars = ax1.bar(filtered['number_inpatient'], filtered['total'], color=color_bar, alpha=0.35, width=0.55, label='Encounter Volume')
    ax1.set_xlabel('Prior Inpatient Visits in Preceding 12 Months', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Total Encounter Volume (Log Scale)', fontsize=12, fontweight='bold', color=color_bar)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color_bar)

    for bar, count in zip(bars, filtered['total']):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, count * 1.15, f'{count:,}', ha='center', va='bottom', fontsize=9, color=color_bar, fontweight='bold')

    # Secondary line axis for rate
    ax2 = ax1.twinx()
    color_line = '#c0392b'
    line = ax2.plot(filtered['number_inpatient'], filtered['rate_30d'], color=color_line, marker='o', linewidth=2.8, markersize=8, label='30-Day Readmission Rate (%)')
    ax2.set_ylabel('30-Day Readmission Rate (%)', fontsize=12, fontweight='bold', color=color_line)
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim(0, 42)

    for x, y in zip(filtered['number_inpatient'], filtered['rate_30d']):
        ax2.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color=color_line)

    plt.title('SWYNEX Technologies — Prior Inpatient Admissions as Primary Readmission Predictor\n(Escalation from 8.44% at 0 Visits to 34.58% at 6 Visits)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(output_dir, '03_prior_inpatient_utilization.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_discharge_disposition_figure(df: pd.DataFrame, output_dir: str):
    """Figure 4: Discharge Disposition vs Readmission Risk."""
    top_disch = df['discharge_disposition_desc'].value_counts().head(7).index
    sub = df[df['discharge_disposition_desc'].isin(top_disch)]

    rates = sub.groupby('discharge_disposition_desc').agg(
        total=('encounter_id', 'count'),
        rate_30d=('readmit_30d', lambda x: x.mean() * 100),
        rate_any=('readmit_any', lambda x: x.mean() * 100)
    ).sort_values(by='rate_30d', ascending=True).reset_index()

    # Shorten labels for clean display
    label_map = {
        'Expired': 'Expired (Deceased) [0.0%]',
        'Discharged to home': 'Home [9.3%]',
        'Discharged/transferred to home with home health service': 'Home w/ Health Service [12.7%]',
        'Not Available / Unmapped': 'Unmapped / Unknown [12.4%]',
        'Discharged/transferred to SNF': 'Skilled Nursing Facility [14.7%]',
        'Discharged/transferred to another short term hospital': 'Short Term Hospital [16.1%]',
        'Discharged/transferred to another rehab fac including rehab units of a hospital .': 'Rehab Facility [27.7%]'
    }
    rates['Display_Label'] = rates['discharge_disposition_desc'].map(label_map).fillna(rates['discharge_disposition_desc'])

    plt.figure(figsize=(12, 6.5))
    bars = plt.barh(rates['Display_Label'], rates['rate_30d'], color=['#7f8c8d', '#27ae60', '#2980b9', '#34495e', '#e67e22', '#d35400', '#c0392b'], edgecolor='black', linewidth=0.5)

    plt.title('SWYNEX Technologies — 30-Day Readmission Risk Across Top Discharge Dispositions', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('30-Day Readmission Rate (%)', fontsize=11, fontweight='semibold')
    plt.ylabel('Discharge Destination', fontsize=11, fontweight='semibold')
    plt.xlim(0, 32)

    for bar, n in zip(bars, rates['total']):
        w = bar.get_width()
        plt.text(w + 0.6, bar.get_y() + bar.get_height() / 2.0, f'{w:.2f}% (N={n:,})', va='center', ha='left', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(output_dir, '04_discharge_disposition_risk.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_correlation_heatmap_figure(df: pd.DataFrame, output_dir: str):
    """Figure 5: Spearman Correlation Matrix Heatmap."""
    num_cols = ['time_in_hospital', 'num_medications', 'num_procedures', 'num_lab_procedures',
                'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient', 'age_midpoint']

    labels = ['LOS (Days)', 'Medications', 'Procedures', 'Lab Tests', 'Diagnoses', 'Inpatient Visits', 'Emergency Visits', 'Outpatient Visits', 'Age Midpoint']

    corr = df[num_cols].corr(method='spearman')

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.6,
        vmin=-0.2,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.6,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation Coefficient"}
    )

    plt.title('SWYNEX Technologies — Clinical Features Spearman Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(output_dir, '05_correlation_heatmap.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def generate_hba1c_and_medchange_figure(df: pd.DataFrame, output_dir: str):
    """Figure 6: Glycemic Testing (HbA1c) & Medication Change Dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: HbA1c result testing
    a1c_rates = df.groupby('A1Cresult').agg(
        total=('encounter_id', 'count'),
        rate_30d=('readmit_30d', lambda x: x.mean() * 100)
    ).reindex(['Norm', '>7', '>8', 'Not Tested']).reset_index()

    bars_a = axes[0].bar(a1c_rates['A1Cresult'], a1c_rates['rate_30d'], color=['#27ae60', '#2980b9', '#e67e22', '#c0392b'], width=0.5, edgecolor='black', linewidth=0.5)
    axes[0].set_title('A: 30-Day Readmission by HbA1c Lab Result Category', fontsize=11, fontweight='bold', pad=12)
    axes[0].set_xlabel('HbA1c Testing Status', fontsize=10, fontweight='semibold')
    axes[0].set_ylabel('30-Day Readmission Rate (%)', fontsize=10, fontweight='semibold')
    axes[0].set_ylim(0, 15)

    for bar, n in zip(bars_a, a1c_rates['total']):
        y = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, y + 0.35, f'{y:.2f}%\n(N={n:,})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel B: Medication Change Status
    chg_rates = df.groupby('change').agg(
        total=('encounter_id', 'count'),
        rate_30d=('readmit_30d', lambda x: x.mean() * 100),
        rate_any=('readmit_any', lambda x: x.mean() * 100)
    ).reset_index()
    chg_rates['Label'] = chg_rates['change'].map({'Ch': 'Dosage Changed (Ch)', 'No': 'No Change (No)'})

    bars_b = axes[1].bar(chg_rates['Label'], chg_rates['rate_30d'], color=['#e67e22', '#2980b9'], width=0.45, edgecolor='black', linewidth=0.5)
    axes[1].set_title('B: 30-Day Readmission by Diabetes Medication Change', fontsize=11, fontweight='bold', pad=12)
    axes[1].set_xlabel('Medication Change Status During Admission', fontsize=10, fontweight='semibold')
    axes[1].set_ylabel('30-Day Readmission Rate (%)', fontsize=10, fontweight='semibold')
    axes[1].set_ylim(0, 15)

    for bar, n, r_any in zip(bars_b, chg_rates['total'], chg_rates['rate_any']):
        y = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, y + 0.35, f'30d: {y:.2f}%\nAny: {r_any:.1f}%\n(N={n:,})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('SWYNEX Technologies — Glycemic Management & Pharmacotherapy Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, '06_hba1c_and_medchange.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataset()
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    generate_target_distribution_figure(df, OUTPUT_DIR)
    generate_numerical_distributions_figure(df, OUTPUT_DIR)
    generate_prior_inpatient_figure(df, OUTPUT_DIR)
    generate_discharge_disposition_figure(df, OUTPUT_DIR)
    generate_correlation_heatmap_figure(df, OUTPUT_DIR)
    generate_hba1c_and_medchange_figure(df, OUTPUT_DIR)
    print("All 6 publication-quality figures successfully exported to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
