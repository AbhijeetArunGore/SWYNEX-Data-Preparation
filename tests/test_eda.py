"""Unit test suite for SWYNEX EDA utility module (src/eda.py)."""
import pytest
import numpy as np
import pandas as pd
from src.eda import (
    profile_dataset,
    compute_numerical_summary,
    compute_categorical_summary,
    compute_bivariate_rates,
    group_continuous_stats,
    perform_mann_whitney_u,
    perform_chi_square,
    compute_correlation_matrix
)


@pytest.fixture
def sample_clinical_df():
    """Create a controlled synthetic clinical dataset for testing EDA functions."""
    return pd.DataFrame({
        'encounter_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'patient_nbr': [101, 102, 101, 103, 104, 105, 102, 106, 107, 108],
        'time_in_hospital': [2, 5, 3, 7, 1, 4, 6, 2, 8, 3],
        'num_medications': [10, 15, 12, 20, 8, 14, 18, 9, 22, 11],
        'number_inpatient': [0, 1, 0, 2, 0, 0, 1, 0, 3, 0],
        'age': ['[50-60)', '[60-70)', '[50-60)', '[70-80)', '[40-50)', '[60-70)', '[60-70)', '[50-60)', '[80-90)', '[40-50)'],
        'readmitted': ['NO', '>30', 'NO', '<30', 'NO', '>30', '<30', 'NO', '<30', 'NO']
    })


def test_profile_dataset(sample_clinical_df):
    metrics = profile_dataset(sample_clinical_df)
    assert metrics['total_rows'] == 10
    assert metrics['total_columns'] == 7
    assert metrics['unique_patients'] == 8
    assert metrics['repeat_admission_pct'] == 20.0
    assert metrics['total_null_values'] == 0


def test_compute_numerical_summary(sample_clinical_df):
    summary = compute_numerical_summary(sample_clinical_df, columns=['time_in_hospital', 'num_medications'])
    assert len(summary) == 2
    los_row = summary[summary['Feature'] == 'time_in_hospital'].iloc[0]
    assert los_row['Count'] == 10
    assert los_row['Mean'] == 4.1
    assert los_row['Min'] == 1.0
    assert los_row['Max'] == 8.0


def test_compute_categorical_summary(sample_clinical_df):
    cat_summary = compute_categorical_summary(sample_clinical_df, columns=['age', 'readmitted'])
    assert len(cat_summary) == 2
    readmit_row = cat_summary[cat_summary['Feature'] == 'readmitted'].iloc[0]
    assert readmit_row['Top_Category'] == 'NO'
    assert readmit_row['Top_Frequency'] == 5
    assert readmit_row['Top_Percentage'] == 50.0


def test_compute_bivariate_rates(sample_clinical_df):
    rates = compute_bivariate_rates(sample_clinical_df, group_col='number_inpatient', target_col='readmitted')
    assert not rates.empty
    zero_row = rates[rates['Group_Value'] == '0'].iloc[0]
    assert zero_row['Total_Encounters'] == 6
    assert zero_row['Readmit_30d_Count'] == 0
    assert zero_row['Readmit_30d_Rate_Pct'] == 0.0


def test_group_continuous_stats(sample_clinical_df):
    stats_df = group_continuous_stats(sample_clinical_df, continuous_col='time_in_hospital', group_col='readmitted')
    assert len(stats_df) == 3
    no_stats = stats_df[stats_df['Group'] == 'NO'].iloc[0]
    assert no_stats['Count'] == 5


def test_perform_mann_whitney_u(sample_clinical_df):
    res = perform_mann_whitney_u(sample_clinical_df, continuous_col='time_in_hospital', group_col='readmitted', group_val1='<30', group_val2='NO')
    assert 'u_statistic' in res
    assert 'p_value' in res
    assert isinstance(res['statistically_significant'], bool)


def test_perform_chi_square(sample_clinical_df):
    res = perform_chi_square(sample_clinical_df, cat_col1='age', cat_col2='readmitted')
    assert 'chi2_statistic' in res
    assert 'p_value' in res
    assert 'cramers_v' in res


def test_compute_correlation_matrix(sample_clinical_df):
    corr = compute_correlation_matrix(sample_clinical_df, columns=['time_in_hospital', 'num_medications'])
    assert corr.shape == (2, 2)
    assert corr.loc['time_in_hospital', 'time_in_hospital'] == 1.0
