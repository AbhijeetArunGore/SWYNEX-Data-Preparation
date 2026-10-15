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
