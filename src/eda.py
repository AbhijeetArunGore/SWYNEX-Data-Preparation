"""SWYNEX Exploratory Data Analysis (EDA) Utility Module.

Provides modular, reproducible functions for clinical dataset profiling,
descriptive statistics computation, bivariate outcome analysis, statistical
hypothesis testing, and visualization generation.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def profile_dataset(df: pd.DataFrame) -> Dict[str, Union[int, float, List[str]]]:
    """Generate high-level structural and integrity metrics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input clinical dataset.

    Returns
    -------
    Dict[str, Union[int, float, List[str]]]
        Dictionary containing row count, column count, memory usage in MB,
        duplicate row count, unique patient count, and total null values.
    """
    total_rows = len(df)
    unique_patients = df['patient_nbr'].nunique() if 'patient_nbr' in df.columns else total_rows
    repeat_rate = float((1.0 - (unique_patients / total_rows)) * 100.0) if total_rows > 0 else 0.0

    return {
        'total_rows': total_rows,
        'total_columns': df.shape[1],
        'memory_mb': round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 2),
        'exact_duplicates': int(df.duplicated().sum()),
        'unique_patients': int(unique_patients),
        'repeat_admission_pct': round(repeat_rate, 2),
        'total_null_values': int(df.isnull().sum().sum()),
        'column_names': df.columns.tolist()
    }


def compute_numerical_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics for continuous/discrete variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input clinical dataset.
    columns : Optional[List[str]], default=None
        List of numerical columns to evaluate. If None, auto-selects numeric types.

    Returns
    -------
    pd.DataFrame
        Table containing count, mean, standard deviation, median, IQR, min, max,
        and skewness for each evaluated numerical feature.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    records = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = float(q75 - q25)
        skew = float(series.skew()) if len(series) > 2 else 0.0

        records.append({
            'Feature': col,
            'Count': int(series.count()),
            'Mean': round(float(series.mean()), 2),
            'Std': round(float(series.std()), 2),
            'Median': round(float(series.median()), 2),
            'IQR': round(iqr, 2),
            'Min': round(float(series.min()), 2),
            'Max': round(float(series.max()), 2),
            'Skewness': round(skew, 2)
        })

    return pd.DataFrame(records)


def compute_categorical_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    top_n: int = 3
) -> pd.DataFrame:
    """Compute descriptive frequency metrics for categorical variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input clinical dataset.
    columns : Optional[List[str]], default=None
        Categorical columns to evaluate.
    top_n : int, default=3
        Number of most frequent categories to list.

    Returns
    -------
    pd.DataFrame
        Summary table with unique count, top category name, top frequency,
        and percentage of occurrences.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    records = []
    total_len = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        n_unique = series.nunique(dropna=False)
        val_counts = series.value_counts(dropna=False)
        top_cat = str(val_counts.index[0]) if len(val_counts) > 0 else 'N/A'
        top_freq = int(val_counts.iloc[0]) if len(val_counts) > 0 else 0
        top_pct = round((top_freq / total_len) * 100, 2) if total_len > 0 else 0.0

        records.append({
            'Feature': col,
            'Unique_Categories': int(n_unique),
            'Top_Category': top_cat,
            'Top_Frequency': top_freq,
            'Top_Percentage': top_pct
        })

    return pd.DataFrame(records)


def compute_bivariate_rates(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = 'readmitted',
    min_count: int = 1
) -> pd.DataFrame:
    """Calculate encounter counts and readmission rates across categorical levels.

    Parameters
    ----------
    df : pd.DataFrame
        Input clinical dataset.
    group_col : str
        Feature name to group by.
    target_col : str, default='readmitted'
        Outcome column containing categories such as '<30', '>30', 'NO'.
    min_count : int, default=1
        Minimum group sample size threshold for filtering sparse levels.

    Returns
    -------
    pd.DataFrame
        Table with Group_Value, Total_Encounters, Cohort_Percentage,
        Readmit_30d_Count, Readmit_30d_Rate_Pct, Any_Readmit_Rate_Pct.
    """
    if group_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Columns '{group_col}' or '{target_col}' not found in DataFrame.")

    total_encounters = len(df)
    grouped = df.groupby(group_col, observed=True)

    records = []
    for val, grp in grouped:
        n = len(grp)
        if n < min_count:
            continue
        c_30d = int((grp[target_col] == '<30').sum())
        c_any = int((grp[target_col].isin(['<30', '>30'])).sum())
        rate_30d = round((c_30d / n) * 100.0, 2) if n > 0 else 0.0
        rate_any = round((c_any / n) * 100.0, 2) if n > 0 else 0.0
        cohort_pct = round((n / total_encounters) * 100.0, 2) if total_encounters > 0 else 0.0

        records.append({
            'Group_Value': str(val),
            'Total_Encounters': n,
            'Cohort_Percentage': cohort_pct,
            'Readmit_30d_Count': c_30d,
            'Readmit_30d_Rate_Pct': rate_30d,
            'Any_Readmit_Rate_Pct': rate_any
        })

    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df = out_df.sort_values(by='Total_Encounters', ascending=False).reset_index(drop=True)
    return out_df


def group_continuous_stats(
    df: pd.DataFrame,
    continuous_col: str,
    group_col: str
) -> pd.DataFrame:
    """Calculate distributional statistics of a continuous variable grouped by an attribute.

    Parameters
    ----------
    df : pd.DataFrame
        Input clinical dataset.
    continuous_col : str
        Numerical column to evaluate.
    group_col : str
        Categorical grouping column.

    Returns
    -------
    pd.DataFrame
        Aggregated statistics per group including count, mean, std, median, IQR.
    """
    if continuous_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Requested columns not present in DataFrame.")

    records = []
    for val, grp in df.groupby(group_col, observed=True):
        s = grp[continuous_col].dropna()
        q25 = float(s.quantile(0.25)) if len(s) > 0 else 0.0
        q75 = float(s.quantile(0.75)) if len(s) > 0 else 0.0

        records.append({
            'Group': str(val),
            'Count': int(len(s)),
            'Mean': round(float(s.mean()), 2) if len(s) > 0 else 0.0,
            'Std': round(float(s.std()), 2) if len(s) > 1 else 0.0,
            'Median': round(float(s.median()), 2) if len(s) > 0 else 0.0,
            'IQR': round(q75 - q25, 2),
            'Min': round(float(s.min()), 2) if len(s) > 0 else 0.0,
            'Max': round(float(s.max()), 2) if len(s) > 0 else 0.0
        })

    return pd.DataFrame(records)

