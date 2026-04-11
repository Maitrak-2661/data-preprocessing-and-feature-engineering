<div align="center">

# 🏦 Holistic Data Preparer
### *End-to-End Data Preprocessing & Feature Engineering Pipeline for Customer Credit Risk Analysis*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

> **A comprehensive, real-world data preparation pipeline** that merges multi-source financial data, handles missing values, removes outliers, engineers meaningful features, and produces a clean, model-ready dataset — all applied to a customer loan default prediction problem.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset Sources](#-dataset-sources)
- [Project Structure](#-project-structure)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
  - [Part A — Conceptual Foundation](#part-a--conceptual-foundation)
  - [Part B — Data Acquisition](#part-b--data-acquisition)
  - [Part C — Data Understanding & Cleaning](#part-c--data-understanding--cleaning)
  - [Part D — Outlier Handling](#part-d--outlier-handling)
  - [Part E — Feature Engineering](#part-e--feature-engineering)
  - [Part F — Feature Scaling](#part-f--feature-scaling)
  - [Part G — Feature Construction & Transformation](#part-g--feature-construction--transformation)
  - [Part H — Final Deliverable](#part-h--final-deliverable)
- [Key Libraries](#-key-libraries)
- [How to Run](#-how-to-run)
- [Output](#-output)
- [Results Summary](#-results-summary)

---

## 🔍 Overview

This project simulates a **real-world data science workflow** for a financial institution. The goal is to take raw, messy, multi-source data about customers and loans — and transform it into a **clean, feature-rich dataset** ready for machine learning models such as loan default prediction.

```
Raw Multi-Source Data  ──►  Merged Dataset  ──►  Cleaned & Imputed  ──►  Engineered Features  ──►  Model-Ready CSV
   (CSV, JSON, SQL, API)        (1000 rows)          (Missing values,       (Encoding, Scaling,       (842 rows,
                                                       outliers handled)      Transformations)          32 features)
```

---

## 📂 Dataset Sources

The project integrates data from **4 different sources**, simulating how data arrives in real enterprise environments:

| File | Source Type | Contents | Records |
|------|-------------|----------|---------|
| `transactions.csv` | CSV File | Loan amounts, purpose, transaction count, spending ratio | 1,000 rows |
| `customer_metadata.json` | JSON File | Customer age, gender, region, education, employment type | 1,000 rows |
| `loan_repayment.db` | SQLite Database | Loan details, repayment history, credit scores, income | 1,000 rows |
| `economic_indicators_api.json` | Simulated API | Customer join date, default flag | 1,000 rows |

> All four sources are joined on `customer_id` to form a single unified dataset of **14 features** before processing begins.

---

## 📁 Project Structure

```
final pr/
│
├── 📓 fixed_Holistic_data_preparer.ipynb   ← Main notebook (full pipeline)
├── 📊 transactions.csv                      ← Raw transaction data (CSV source)
├── 🧾 customer_metadata.json               ← Customer demographics (JSON source)
├── 🗄️  loan_repayment.db                   ← Loan & repayment data (SQLite source)
├── 🌐 economic_indicators_api.json         ← API-simulated default flags
├── 📈 data_profiling_report.html           ← Auto-generated EDA profiling report
└── ✅ final_cleaned_data.csv               ← Final output: 842 rows × 32 features
```

---

## 🔄 Pipeline Walkthrough

### Part A — Conceptual Foundation

Before any code is written, this section establishes the **theory** behind the pipeline:

- What is Data Analysis and why it matters for financial risk
- How to plan a data science project end-to-end
- Understanding **tensors** (scalars, vectors, matrices) as the foundation for numerical data manipulation

---

### Part B — Data Acquisition

The pipeline demonstrates how to **load data from multiple real-world source types**:

```python
# CSV
df_csv = pd.read_csv("transactions.csv")

# JSON
df_json = pd.read_json("customer_metadata.json")

# SQL Database
conn = sqlite3.connect("loan_repayment.db")
df_sql = pd.read_sql_query("SELECT * FROM loan_repayment_history", conn)

# Simulated API (JSON)
with open("economic_indicators_api.json", 'r') as f:
    api_data = json.load(f)
df_api = pd.json_normalize(api_data['records'])
```

All four DataFrames are **merged on `customer_id`**, and invalid string representations of null values (`"None"`, `"NULL"`, `"N/A"`, `" "`, `"-"`) are standardized to `NaN`.

---

### Part C — Data Understanding & Cleaning

**Exploratory analysis** is performed using `.info()`, `.describe()`, and `.isnull().sum()`. A rich **automated profiling report** is generated:

```python
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Customer Credit Risk - Data Profiling Report")
profile.to_file("data_profiling_report.html")
```

**Missing value imputation** strategies compared:

| Strategy | Best For |
|----------|----------|
| `SimpleImputer(strategy='mean')` | Numerical columns with symmetric distribution |
| `SimpleImputer(strategy='most_frequent')` | Categorical columns (gender, employment_type) |
| `KNNImputer` | Leverages similar records to estimate missing values |
| `IterativeImputer` (MICE) | Complex relationships between features |
| Random Sample Imputation | Preserves original distribution |
| Complete Case Analysis | Baseline comparison — drops all rows with nulls |

> ✅ **Decision**: `KNNImputer` and `IterativeImputer` were selected as they preserve data relationships better than simply dropping rows (CCA lost ~15% of data).

---

### Part D — Outlier Handling

Outliers in `age`, `annual_income`, `loan_amount`, `credit_score`, and `transaction_count` are identified and treated using four methods:

| Method | Approach |
|--------|----------|
| **Z-Score** | Removes records where Z > 3 (assumes normal distribution) |
| **IQR (Interquartile Range)** | Removes records outside `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR` |
| **Percentile Capping** | Removes values below 5th and above 95th percentile |
| **Winsorization** | Clips extreme values to the boundary instead of removing them |

> ✅ **Decision**: `IQR method` was chosen for downstream processing — it effectively removes outliers while preserving the most data.

---

### Part E — Feature Engineering

**Categorical Encoding:**

| Variable | Encoding Method | Reason |
|----------|----------------|--------|
| `education_level` | Ordinal Encoding | Natural order: Primary → Secondary → Graduate → Post-Graduate |
| `employment_type`, `gender` | Label Encoding | Binary or low-cardinality categories |
| `region`, `loan_purpose` | One-Hot Encoding (drop first) | No ordinal relationship between categories |

**Date/Time Features** extracted from `join_date`:
- `year`, `month`, `day`, `weekday`

**Numerical Binning:**

| Variable | Method | Bins |
|----------|--------|------|
| `annual_income`, `repayment_history` | Uniform Binning (KBinsDiscretizer) | 4 bins |
| `transaction_count` | Quantile Binning | 4 bins |
| `transaction_count` | K-Means Binning | 4 bins |
| `credit_score` | Binarization | ≥ 700 = 1 (good), < 700 = 0 (poor) |

---

### Part F — Feature Scaling

Multiple scaling strategies are applied and compared on numerical features:

| Scaler | Formula | Best For |
|--------|---------|----------|
| `StandardScaler` | `(x - μ) / σ` | Normally distributed features |
| `MinMaxScaler` | `(x - min) / (max - min)` | Bounded range needed [0, 1] |
| `MaxAbsScaler` | `x / max(|x|)` | Sparse data |
| `RobustScaler` | `(x - median) / IQR` | Data with remaining outliers |

---

### Part G — Feature Construction & Transformation

**Mathematical Transformations** applied to skewed features:

| Transformation | Applied To | Purpose |
|---------------|-----------|---------|
| Log Transform (`np.log1p`) | `spending_ratio` | Reduce right skew |
| Reciprocal Transform | `spending_ratio` | Compress extreme values |
| Square Root Transform | `spending_ratio` | Moderate skew reduction |
| Box-Cox Transform | `loan_amount`, `annual_income` | Normalize positive distributions |
| Yeo-Johnson Transform | `loan_amount`, `annual_income` | Like Box-Cox but handles zeros/negatives |

**Constructed Features** (domain-driven):

```python
# Ratio of loan obligation to income — key credit risk indicator
df['debt_to_income_ratio'] = df['loan_amount'] / df['annual_income']

# Average monthly transaction activity
df['average_monthly_transactions'] = df['transaction_count'] / months_since_join

# Spending relative to income
df['spending_to_income_ratio'] = df['spending_ratio'] / df['annual_income']
```

---

### Part H — Final Deliverable

The pipeline outputs `final_cleaned_data.csv` — a fully processed dataset ready for model training.

| Metric | Before | After |
|--------|--------|-------|
| Rows | 1,000 | **842** |
| Columns | 14 (raw) | **32** (engineered) |
| Missing Values | Present | **0** |
| Outliers | Present | **Handled** |
| Categorical Variables | Raw strings | **Encoded numerically** |
| New Features Added | — | **3 domain features** |

---

## 📦 Key Libraries

```python
import pandas as pd               # Data manipulation
import numpy as np                # Numerical operations
import matplotlib.pyplot as plt   # Visualization
import seaborn as sns             # Statistical plots
import sqlite3                    # SQL database connection
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    OrdinalEncoder, LabelEncoder, OneHotEncoder,
    KBinsDiscretizer, Binarizer,
    StandardScaler, MinMaxScaler, MaxAbsScaler,
    RobustScaler, Normalizer, FunctionTransformer, PowerTransformer
)
from ydata_profiling import ProfileReport  # Automated EDA
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Maitrak-2661/data-preprocessing-and-feature-engineering.git
cd "data-preprocessing-and-feature-engineering/final pr"
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook fixed_Holistic_data_preparer.ipynb
```

### 4. Run all cells
In Jupyter: **Kernel → Restart & Run All**

> ⚠️ Make sure all data files (`transactions.csv`, `customer_metadata.json`, `loan_repayment.db`, `economic_indicators_api.json`) are in the **same directory** as the notebook before running.

---

## 📤 Output

After running the notebook, you will find:

| Output File | Description |
|-------------|-------------|
| `final_cleaned_data.csv` | 842 rows × 32 features, fully preprocessed and model-ready |
| `data_profiling_report.html` | Interactive EDA report — open in any browser |

---

## 📊 Results Summary

| Pipeline Stage | Technique Applied | Outcome |
|---------------|------------------|---------|
| Data Acquisition | CSV, JSON, SQLite, API merge | 1 unified DataFrame |
| Missing Value Handling | KNN Imputer, MICE, Simple Imputer | 0 missing values |
| Outlier Removal | IQR Method | Cleaner distributions |
| Encoding | Ordinal, Label, One-Hot | All categoricals → numeric |
| Date Features | Datetime extraction | 4 new temporal columns |
| Binning | Uniform, Quantile, K-Means | Discretized continuous vars |
| Scaling | StandardScaler, MinMaxScaler, RobustScaler | Multiple scale options explored |
| Transformation | Log, Box-Cox, Yeo-Johnson | Normalized skewed features |
| Feature Construction | DTI ratio, spending index | 3 new domain features |

---

<div align="center">

**Made with 🧠 + 📊 for the Data Science community**

*Contributions and feedback are welcome!*

</div>
