import pytest
import numpy as np
import pandas as pd
from src.utils import map_icd9_category, convert_age_to_midpoint, normalize_lab_tests

def test_map_icd9_category():
    assert map_icd9_category('414') == 'Circulatory'
    assert map_icd9_category('486') == 'Respiratory'
    assert map_icd9_category('250.02') == 'Diabetes'
    assert map_icd9_category('V45') == 'Other'
    assert map_icd9_category('?') == 'Missing/Unknown'

def test_convert_age_to_midpoint():
    assert convert_age_to_midpoint('[50-60)') == 55.0
    assert convert_age_to_midpoint('[0-10)') == 5.0

def test_normalize_lab_tests():
    df = pd.DataFrame({'max_glu_serum': ['None', np.nan, 'Norm'], 'A1Cresult': ['>8', 'None', np.nan]})
    df_clean = normalize_lab_tests(df)
    assert list(df_clean['max_glu_serum']) == ['Not Tested', 'Not Tested', 'Norm']
    assert list(df_clean['A1Cresult']) == ['>8', 'Not Tested', 'Not Tested']
