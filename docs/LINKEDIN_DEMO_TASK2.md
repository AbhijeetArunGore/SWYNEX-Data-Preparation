# SWYNEX Internship — Task 2: Exploratory Data Analysis (EDA)
## LinkedIn Post Copy & Video Walkthrough Guide

---

### 📝 LinkedIn Post Draft

Excited to share that I have completed **Task 2: Exploratory Data Analysis (EDA)** as part of my Data Science Internship with @SWYNEX Technologies! 🚀

Building directly upon the clean clinical data foundation established in Task 1, this task delivered an in-depth, hypothesis-driven exploratory investigation of the **Diabetes 130-US Hospitals dataset (1999–2008)** from the UCI Machine Learning Repository, analyzing **101,763 inpatient encounters** across **71,515 unique patients** and **54 features**.

🔍 **Key Verified Findings & Data Science Insights**:
1. **Prior Utilization as Primary Risk Signal**: Prior acute inpatient admissions showed the steepest readmission gradient—escalating 30-day early readmissions from **8.44%** (0 visits) to **34.58%** (6 visits; Mann-Whitney $U = 401,064,764, p < 10^{-15}$).
2. **Severe Class Imbalance**: Readmission within 30 days (`<30`) occurs in only **11.16%** of encounters (~8:1 imbalance). A trivial null classifier predicting zero readmission achieves **88.84% accuracy**, confirming that future model evaluation must be strictly governed by **PR-AUC** and calibrated operating thresholds rather than raw accuracy.
3. **Post-Acute Transfer Vulnerability**: Transfers to rehabilitation facilities (**27.71%**) and Skilled Nursing Facilities (**14.66%**) experienced dramatically higher readmission risk than routine home discharges (**9.30%**).
4. **Terminal Event Target Leakage**: Identified 1,642 in-hospital patient deaths (`Expired`) showing an exact **0.00%** readmission rate. Deceased patients are physically incapable of being readmitted; establishing the mandatory rule that these records must be excluded prior to predictive modeling.
5. **Intra-Patient Clustering**: 29.72% of encounters represent repeat hospitalizations by recurring patients (up to 40 visits per individual), demonstrating that future model validation must strictly enforce **GroupKFold** grouped by `patient_nbr` to avoid optimistic data leakage.

💡 **Core Learning**: Exploratory Data Analysis is not just about producing descriptive charts—it is about establishing rigorous statistical governance, auditing data leakage vectors, and translating empirical distributions into actionable modeling constraints.

🛠️ **Tools & Technologies**: Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Jupyter Notebook, Git/GitHub.

📁 **GitHub Repository**: https://github.com/AbhijeetArunGore/SWYNEX-Data-Preparation  
📓 **Task 2 Notebook**: `notebooks/02_exploratory_data_analysis.ipynb`  
📄 **Executive EDA Report**: `reports/eda/eda_summary_report.md`  

#SWYNEX #Internship #DataScience #ExploratoryDataAnalysis #DataAnalytics #HealthcareAnalytics #Python #Pandas #MachineLearning #ClinicalData

---

### 🎥 60–90 Second Screen Recording Video Walkthrough Script

| Time | Visual on Screen | Spoken Script / Narration |
| :---: | :--- | :--- |
| **0:00 - 0:15** | GitHub repository root ([SWYNEX-Data-Preparation](https://github.com/AbhijeetArunGore/SWYNEX-Data-Preparation)), showing the clean folder structure (`notebooks/`, `reports/`, `src/`, `tests/`). | "Hello everyone! Today I'm presenting Task 2 of my Data Science internship at SWYNEX Technologies: Exploratory Data Analysis of hospital encounters for diabetic patients. We built directly upon our Task 1 cleaned dataset of 101,763 inpatient records." |
| **0:15 - 0:35** | Scroll into `notebooks/02_exploratory_data_analysis.ipynb`, showcasing Section 3 data validation and Section 8 target distribution. | "In our notebook, we first validated our 101,763 rows and 54 features with zero unhandled nulls. We audited our primary target—30-day hospital readmission—revealing an 11.16% prevalence. This 8:1 class imbalance demonstrates why overall accuracy is misleading and why PR-AUC must govern our models." |
| **0:35 - 0:55** | Highlight Figure 3 (Prior Inpatient Visits vs Readmission Rate) and Figure 4 (Discharge Disposition Risk). | "Looking at key bivariate drivers, prior inpatient admissions showed the strongest effect size in the dataset, driving 30-day readmission risk from 8.44% at 0 visits up to 34.58% at 6 visits ($p < 10^{-15}$). Furthermore, post-acute transfers to rehab and skilled nursing facilities showed over double the readmission rate of home discharges." |
| **0:55 - 1:15** | Show Section 11 Correlation Heatmap and Section 13 Anti-Leakage Rules. | "Crucially, our EDA identified two major data leakage risks: first, 1,642 deceased patient records that must be filtered out before training; and second, a 29.7% repeat patient admission rate that requires GroupKFold validation on patient IDs." |
| **1:15 - 1:30** | Show `reports/eda/eda_summary_report.md` and repository README. | "The complete, fully executed notebook, executive findings report, and unit test suite are published on my GitHub repository. Thank you for watching, and stay tuned for Task 3!" |
