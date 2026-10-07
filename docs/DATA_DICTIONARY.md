# Clinical Dataset Schema & Data Dictionary

## Raw Features Overview
The raw dataset contains 50 clinical features across 101,766 hospital encounters.

### Key Demographics & Identifiers
- `encounter_id`: Unique identifier of an encounter
- `patient_nbr`: Unique identifier of a patient (used to track repeat admissions)
- `race`: Patient race/ethnicity
- `gender`: Patient gender
- `age`: Patient age bracket in 10-year intervals

### Relational Lookups (IDS_mapping.csv)
- `admission_type_id`: Integer key referencing admission type description
- `discharge_disposition_id`: Integer key referencing discharge disposition description
- `admission_source_id`: Integer key referencing admission source description

### Clinical Measurements & Diagnoses
- `time_in_hospital`: Integer days in hospital (1 to 14)
- `num_lab_procedures`: Number of lab tests performed
- `num_procedures`: Number of non-lab procedures performed
- `num_medications`: Number of distinct medications administered
- `number_outpatient`: Outpatient visits in past year
- `number_emergency`: Emergency room visits in past year
- `number_inpatient`: Inpatient admissions in past year
- `diag_1`, `diag_2`, `diag_3`: Primary, secondary, and tertiary ICD-9 diagnosis codes
- `number_diagnoses`: Number of diagnoses entered
