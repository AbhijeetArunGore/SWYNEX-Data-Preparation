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


def parse_ids_mapping(ids_raw_text: str) -> dict:
    """Parse raw text of IDS_mapping.csv into relational dictionaries."""
    lines = ids_raw_text.split('\n')
    mapping_dicts = {'admission_type_id': {}, 'discharge_disposition_id': {}, 'admission_source_id': {}}
    current_key = None
    for line in lines:
        line_str = line.strip()
        if not line_str or line_str == ',':
            continue
        if line_str.startswith('admission_type_id'):
            current_key = 'admission_type_id'
            continue
        elif line_str.startswith('discharge_disposition_id'):
            current_key = 'discharge_disposition_id'
            continue
        elif line_str.startswith('admission_source_id'):
            current_key = 'admission_source_id'
            continue
            
        parts = line_str.split(',', 1)
        if len(parts) == 2 and parts[0].isdigit():
            v_id = int(parts[0])
            desc = parts[1].strip('"').strip()
            if desc.upper() in ['NULL', 'NAN', 'NONE', '']:
                desc = 'Not Available / Unmapped'
            mapping_dicts[current_key][v_id] = desc
    return mapping_dicts


def convert_age_to_midpoint(age_str: str) -> float:
    """Convert age bracket string to numerical midpoint."""
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0, '[40-50)': 45.0,
        '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0, '[80-90)': 85.0, '[90-100)': 95.0
    }
    return age_map.get(str(age_str).strip(), np.nan)
