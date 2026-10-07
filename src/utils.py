"""Preprocessing utility functions for SWYNEX Data Preparation."""
import pandas as pd
import numpy as np

def normalize_lab_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize lab test missing values to 'Not Tested'."""
    df = df.copy()
    for col in ['max_glu_serum', 'A1Cresult']:
        if col in df.columns:
            df[col] = df[col].fillna('Not Tested').replace('None', 'Not Tested')
    return df
