<div align="center">

# 🏥 Patient Health Risk Prediction
### *Data Cleaning & Outlier Handling Pipeline for Clinical Data*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<br/>

> A focused, healthcare-domain data preprocessing pipeline that compares **4 imputation strategies** and **3 outlier removal methods** to produce a clean, reliable dataset ready for disease risk classification.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Part A — Missing Value Imputation](#-part-a--missing-value-imputation)
- [Part B — Outlier Detection & Removal](#-part-b--outlier-detection--removal)
- [Part C — Visual Comparison](#-part-c--visual-comparison)
- [Final Conclusions](#-final-conclusions)
- [Before vs After](#-before-vs-after)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)
- [Project Structure](#-project-structure)

---

## 🔍 Overview

This project tackles a real-world **healthcare data quality problem**. The raw dataset contains 10% missing values across 6 features and multiple clinical outliers — both of which can seriously distort machine learning model performance if left untreated.

The goal is to:
1. Apply and compare multiple **imputation strategies** to handle missing values
2. Apply and compare multiple **outlier removal methods** to clean clinical readings
3. Produce a **final clean dataset** ready for disease risk prediction modelling

```
Raw Patient Data (1,000 rows, 10% missing)
        │
        ▼
┌─────────────────────┐
│  PART A: IMPUTATION │  SimpleImputer · Random Sample · KNNImputer · MICE ✅
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────┐
│  PART B: OUTLIER REMOVAL │  Z-Score · IQR ✅ · Percentile
└─────────┬────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  PART C: VISUAL COMPARISON          │  Side-by-side box plots of glucose
└─────────┬───────────────────────────┘
          │
          ▼
  final_clean_patient_dataset.csv
  (904 rows · 9 features · 0 missing values)
```

**ML Task:** Binary Classification — `disease_risk` (0 = low risk, 1 = high risk)

---

## 📋 Dataset

### Raw Dataset — `patient_health_dataset.csv`

| Property | Value |
|----------|-------|
| Rows | 1,000 |
| Features | 9 |
| Missing Values | 600 (10% across 6 columns) |
| Target | `disease_risk` (0 / 1) |

### Feature Reference

| Feature | Type | Description | Missing |
|---------|------|-------------|---------|
| `patient_id` | ID | Unique patient identifier | ❌ None |
| `age` | Numeric | Patient age (18 – 79) | ✅ 10% |
| `gender` | Categorical | Male / Female | ✅ 10% |
| `region` | Categorical | East / West / North / South | ✅ 10% |
| `bmi` | Numeric | Body Mass Index (8 – 67) | ✅ 10% |
| `blood_pressure` | Numeric | Blood pressure (75 – 228) | ❌ None |
| `cholesterol` | Numeric | Cholesterol level (70 – 420) | ✅ 10% |
| `glucose` | Numeric | Blood glucose level (8 – 295) | ✅ 10% |
| `disease_risk` | Binary | 🎯 Target: 0 = low, 1 = high | ❌ None |

### Statistical Summary (Raw Numerical Features)

| Feature | Mean | Std | Min | Median | Max |
|---------|------|-----|-----|--------|-----|
| `age` | 49.66 | 18.18 | 18 | 50 | 79 |
| `bmi` | 25.46 | 5.95 | 8.06 | 25.20 | 66.90 |
| `blood_pressure` | 122.04 | 18.84 | 75.70 | 121.08 | 228.11 |
| `cholesterol` | 202.68 | 46.59 | 70.42 | 199.33 | 419.91 |
| `glucose` | 102.35 | 33.40 | 8.57 | 100.93 | 295.43 |

> Notable extreme values: BMI max of 66.90, glucose max of 295.43, blood pressure max of 228.11 — all clinical outliers worth addressing.

---

## 🔄 Project Workflow

```
Part A ──► Part B ──► Part C ──► Final Output
Impute     Outliers   Visualize  CSV Export
```

The pipeline is structured in 3 parts inside `patient_health.ipynb` (16 cells total).

---

## 🧹 Part A — Missing Value Imputation

### The Problem

6 out of 9 features each have exactly **100 missing values (10%)**:
`age`, `gender`, `region`, `bmi`, `cholesterol`, `glucose`

> `blood_pressure`, `patient_id`, and `disease_risk` are complete.

### Strategy 1 — SimpleImputer

```python
mean_simple   = SimpleImputer(strategy='mean')
median_simple = SimpleImputer(strategy='median')
mode_simple   = SimpleImputer(strategy='most_frequent')

data_simple[['gender','region']] = mode_simple.fit_transform(...)   # categorical
data_simple['bmi']               = mean_simple.fit_transform(...)    # numeric (mean)
data_simple[['age','cholesterol','glucose']] = median_simple.fit_transform(...)  # numeric (median)
```

**Pros:** Fast, simple, no encoding needed  
**Cons:** Ignores relationships between features; can distort distributions

---

### Strategy 2 — Random Sample Imputation + Missing Indicator

```python
# Flags which values were originally missing
indicator = MissingIndicator(features='missing-only')
indicator_df = pd.DataFrame(indicator.fit_transform(data), columns=indicator_cols)

# Fills each null by randomly sampling from observed values in that column
for col in columns_to_impute:
    imputed_df = random_sample_imp(imputed_df, col)

data_random = pd.concat([imputed_df, indicator_df], axis=1)
```

**Pros:** Preserves the original distribution; `MissingIndicator` creates binary flag columns capturing *which* values were imputed — useful signal for some ML models  
**Cons:** Imputes independently per column; doesn't leverage inter-feature relationships

---

### Strategy 3 — KNN Imputer (k=5, distance-weighted)

```python
# Encode categoricals to numeric before KNN
df_knn['gender'] = df_knn['gender'].map({'Male': 0, 'Female': 1})
df_knn['region'] = df_knn['region'].map({'East': 0, 'West': 1, 'North': 2, 'South': 3})

imputer = KNNImputer(n_neighbors=5, weights='distance')
df_knn.iloc[:, :] = imputer.fit_transform(df_knn)

# Decode back to original labels
df_knn['gender'] = df_knn['gender'].round().map({0: 'Male', 1: 'Female'})
df_knn['region'] = df_knn['region'].round().map({0: 'East', 1: 'West', 2: 'North', 3: 'South'})
```

**How it works:** For each missing value, finds the 5 most similar patients (by Euclidean distance across all features) and fills using a distance-weighted average  
**Pros:** Considers feature relationships; handles both numerical and categorical (with encoding)  
**Cons:** Slower on large datasets; sensitive to scale differences

---

### Strategy 4 — MICE / Iterative Imputer ✅ Selected

```python
mice_imputer = IterativeImputer(max_iter=250, random_state=18)
df_mice.iloc[:, :] = mice_imputer.fit_transform(df_mice)
```

**How it works:** Runs 250 iterations — each feature with missing values is modelled as a function of all other features using regression. Values are updated iteratively until convergence.

**Pros:** Most statistically rigorous; captures complex clinical relationships between BMI, glucose, cholesterol, and blood pressure  
**Cons:** Computationally expensive (250 iterations)

> ✅ **MICE selected** for all downstream outlier processing — it best preserves the natural clinical relationships in the data.

### Imputation Strategy Comparison

| Strategy | Handles Relationships | Preserves Distribution | Speed | Selected |
|----------|:--------------------:|:---------------------:|:-----:|:--------:|
| SimpleImputer | ❌ | Partial | ⚡ Fast | — |
| Random Sample + Indicator | ❌ | ✅ Yes | ⚡ Fast | — |
| KNN Imputer (k=5) | ✅ Yes | ✅ Yes | 🐢 Slow | — |
| **MICE (250 iter)** | **✅ Yes** | **✅ Yes** | **🐢 Slowest** | **✅** |

---

## 📐 Part B — Outlier Detection & Removal

All outlier methods are applied to: `age`, `bmi`, `cholesterol`, `glucose`

> Applied on the MICE-imputed dataset (`df_mice`) as the baseline.

### Method 1 — Z-Score (threshold = 3)

```python
threshold = 3
for col in cols:
    z_score = (df_z[col] - df_z[col].mean()) / df_z[col].std()
    df_z = df_z[z_score.abs() < threshold]
```

Removes records where any feature value is more than **3 standard deviations** from the mean. Best suited for normally distributed data.

**Result:** ~970 rows retained — relatively lenient, misses moderate outliers

---

### Method 2 — IQR (Interquartile Range) ✅ Selected

```python
for col in cols:
    q1 = df_iqr[col].quantile(0.25)
    q3 = df_iqr[col].quantile(0.75)
    iqr_val = q3 - q1
    lower_bound = q1 - 1.5 * iqr_val
    upper_bound = q3 + 1.5 * iqr_val
    df_iqr = df_iqr[~((df_iqr[col] < lower_bound) | (df_iqr[col] > upper_bound))]
```

Removes records outside the fence of **Q1 − 1.5×IQR** and **Q3 + 1.5×IQR`. Does not assume normality — robust for medical data.

**Result:** **904 rows retained** — optimal balance

---

### Method 3 — Percentile (5th – 95th)

```python
for col in cols:
    lower_val = df_perc[col].quantile(0.05)
    upper_val = df_perc[col].quantile(0.95)
    df_perc = df_perc[~((df_perc[col] < lower_val) | (df_perc[col] > upper_val))]
```

Removes the bottom 5% and top 5% of each feature. More aggressive than IQR.

**Result:** ~900 rows retained — too much data loss for a 1,000-row medical dataset

---

### Outlier Method Comparison

| Method | Rows Remaining | Data Loss | Assumption | Selected |
|--------|:--------------:|:---------:|------------|:--------:|
| No treatment (MICE baseline) | 1,000 | 0% | — | — |
| Z-Score (threshold = 3) | ~970 | ~3% | Normal distribution | — |
| **IQR** | **904** | **~9.6%** | **None** | **✅** |
| Percentile (5th–95th) | ~900 | ~10%+ | None | — |

> ✅ **IQR selected** — distribution-free, removes genuine extremes (BMI of 67, glucose of 295) without being overly aggressive.

---

## 📊 Part C — Visual Comparison

A **2×2 grid of horizontal box plots** visualises the effect of each outlier method on the `glucose` feature:

```
┌──────────────────────┬──────────────────────┐
│  Original (MICE)     │  Z-Score             │
│  glucose — with      │  glucose — fewer     │
│  extreme outliers    │  extremes removed    │
├──────────────────────┼──────────────────────┤
│  IQR ✅              │  Percentile          │
│  glucose — clean,    │  glucose — tightest  │
│  balanced removal    │  range, most loss    │
└──────────────────────┴──────────────────────┘
```

**Key visual takeaway:** The IQR plot shows a clean, tight interquartile range with no extreme whiskers, while retaining more rows than the Percentile method.

---

## ✅ Final Conclusions

**1. Most Effective Imputation Strategy**  
KNN and MICE were the strongest approaches because they model relationships between variables (e.g., BMI and glucose co-vary clinically). Simple mean/median imputation ignores these dependencies and can introduce bias.

**2. Best Outlier Handling Method**  
IQR performed best overall. Although the Percentile method removed slightly more outliers, it caused higher data loss. IQR provides a better balance — removing extreme values (like BMI of 67 or glucose of 295) while keeping the majority of valid patient records intact.

**3. Dataset Usability Improvement**  
After cleaning, the dataset has zero missing values, no extreme outliers, and a consistent numerical structure — significantly improving reliability for downstream analysis and modelling.

**4. Overall**  
By combining MICE imputation with IQR outlier removal, the final dataset is statistically stable, clinically plausible, and ready for predictive modelling such as disease risk classification.

---

## 📊 Before vs After

| Metric | Raw Dataset | Final Cleaned |
|--------|:-----------:|:-------------:|
| **Rows** | 1,000 | **904** |
| **Features** | 9 | **9** |
| **Missing Values** | ❌ 600 (10%) | **✅ 0** |
| **Outliers** | ❌ Present | **✅ Removed (IQR)** |
| **Imputation** | ❌ None | **✅ MICE (250 iter)** |
| **Disease Risk (0 / 1)** | — | **542 / 362** |
| **ML Ready** | ❌ | **✅** |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Maitrak-2661/data-preprocessing-and-feature-engineering.git
cd "data-preprocessing-and-feature-engineering/pr 2"
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the notebook

```bash
jupyter notebook patient_health.ipynb
```

### 4. Run all cells

In Jupyter: **Kernel → Restart & Run All**

> ⚠️ `patient_health_dataset.csv` must be in the **same directory** as the notebook before running.

---

## 📦 Dependencies

```
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
scikit-learn >= 1.1       # KNNImputer, IterativeImputer, SimpleImputer, MissingIndicator
jupyter >= 1.0
```

---

## 📁 Project Structure

```
pr 2/
├── 📓 patient_health.ipynb              ← Full pipeline notebook (16 cells, 3 parts)
├── patient_health_dataset.csv           ← Raw input: 1,000 rows, 10% missing values
├── final_clean_patient_dataset.csv      ← Cleaned output: 904 rows, 0 missing values
└── README.md                            ← You are here
```

---

<div align="center">

**Part of the [Data Preprocessing & Feature Engineering](https://github.com/Maitrak-2661/data-preprocessing-and-feature-engineering) repository**

Made with 💙 for the Data Science community

</div>
