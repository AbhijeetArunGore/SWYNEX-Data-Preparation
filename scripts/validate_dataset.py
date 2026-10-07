"""SWYNEX Dataset Integrity Validator Script."""
import os
import pandas as pd

def validate():
    csv_path = os.path.join('data', 'processed', 'diabetic_data_clean.csv')
    assert os.path.exists(csv_path), "Clean CSV file does not exist!"
    df = pd.read_csv(csv_path)
    assert len(df) == 101763, f"Expected 101,763 rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, "Clean dataset contains unhandled null values!"
    print("Dataset validation passed successfully: 101,763 rows, 0 nulls.")

if __name__ == '__main__':
    validate()
