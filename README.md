<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&duration=2500&pause=800&color=2F81F7&center=true&vCenter=true&width=800&lines=Data+Preprocessing+%26+Feature+Engineering;3+Projects+%2B+1+Practical+Exam;Clean+Data+%E2%86%92+Smarter+Models" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)

<br/>

> A collection of **3 end-to-end data science projects + 1 Practical Exam** covering the full data preparation lifecycle —
> from raw, messy, multi-source data to clean, engineered, model-ready datasets.
> Projects span real-world domains: **finance**, **healthcare**, **customer analytics**, and **e-commerce**.

<br/>

[![Stars](https://img.shields.io/github/stars/Maitrak-2661/data-preprocessing-and-feature-engineering?style=social)](https://github.com/Maitrak-2661/data-preprocessing-and-feature-engineering)
[![Forks](https://img.shields.io/github/forks/Maitrak-2661/data-preprocessing-and-feature-engineering?style=social)](https://github.com/Maitrak-2661/data-preprocessing-and-feature-engineering/fork)

</div>

---

## 📚 Table of Contents

| | Section |
|--|---------|
| 🗂️ | [Repository Overview](#️-repository-overview) |
| 🏦 | [Final PR — Loan Default Prediction Pipeline](#-final-pr--loan-default-prediction-pipeline) |
| 📊 | [PR 1 — Customer Churn Prediction](#-pr-1--customer-churn-prediction) |
| 🏥 | [PR 2 — Patient Health Risk Prediction](#-pr-2--patient-health-risk-prediction) |
| 🛒 | [Practical Exam — Customer Purchase Prediction](#-practical-exam--customer-purchase-prediction) |
| 🧠 | [Techniques Across All Projects](#-techniques-across-all-projects) |
| 🚀 | [Getting Started](#-getting-started) |
| 📦 | [Dependencies](#-dependencies) |

---

## 🗂️ Repository Overview

This repository contains progressive data science projects, each building on core skills in **data acquisition, cleaning, feature engineering, EDA, and machine learning**. All three projects follow the same rigorous pipeline philosophy but apply it to distinct real-world domains.

```
📦 data-preprocessing-and-feature-engineering/
│
├── 📁 final pr/          ← Multi-source loan default pipeline  (most advanced)
├── 📁 pr 1/              ← Customer churn analysis & ML
├── 📁 pr 2/              ← Patient health risk preprocessing
└── 📁 practical exam/    ← 🎓 E-commerce purchase prediction (exam project)
```

### At a Glance

| | Project | Domain | Data Sources | Raw Rows | Clean Rows | Features | ML Task |
|--|---------|--------|-------------|----------|------------|----------|---------|
| 🏦 | **Final PR** | Finance / Credit Risk | CSV + JSON + SQL + API | 1,000 | 842 | 32 | Binary Classification |
| 📊 | **PR 1** | Customer Analytics | CSV + JSON + SQLite | 1,000+ | — | 8 | Binary Classification |
| 🏥 | **PR 2** | Healthcare | CSV | 1,000 | 904 | 9 | Binary Classification |
| 🛒 | **Practical Exam** | E-Commerce | CSV + JSON + SQL + API | 1,000 | 1,000 | 39 | Binary Classification |

---

## 🏦 Final PR — Loan Default Prediction Pipeline

> **The most comprehensive project in the repo.** Simulates an enterprise-grade data pipeline that ingests from 4 source types, applies a full preprocessing suite, and produces a 32-feature model-ready dataset.

### 📁 Files

```
final pr/
├── 📓 fixed_Holistic_data_preparer.ipynb   ← Full pipeline (53 cells, 8 parts)
├── transactions.csv                         ← Loan & transaction data
├── customer_metadata.json                   ← Customer demographics
├── loan_repayment.db                        ← SQLite: loans + EMI history
├── economic_indicators_api.json             ← Simulated API: join date, default flag
├── data_profiling_report.html               ← Auto-generated EDA report
└── final_cleaned_data.csv                   ← Output: 842 rows × 32 features
```

### 🎯 Objective

Predict whether a customer will **default on a loan** (0 = repaid, 1 = defaulted) using merged, cleaned, and engineered features from four independent data sources.

```
CSV ──┐
JSON──┤──► MERGE ──► IMPUTE ──► OUTLIERS ──► ENCODE ──► SCALE ──► ✅ 842 × 32
SQL ──┤   1,000      KNN/MICE    IQR        OHE/LE     Robust
API ──┘
```

### 🔌 Data Sources

| Source | File | Key Information |
|--------|------|----------------|
| 📄 CSV | `transactions.csv` | Loan amount, purpose, transaction count, spending ratio |
| 🧾 JSON | `customer_metadata.json` | Age, gender, region, education, employment type |
| 🗄️ SQLite | `loan_repayment.db` | Annual income, credit score, repayment history (3 tables, 23K+ rows) |
| 🌐 API | `economic_indicators_api.json` | Join date, default flag |

### ⚙️ Pipeline (8 Parts)

<details>
<summary><b>Part A — Conceptual Foundation</b></summary>

- What is Data Analysis and its role in financial risk
- Planning a data science project end-to-end
- Tensors explained: scalars, vectors, matrices with NumPy
- Problem framing: supervised binary classification, ROC-AUC metric

</details>

<details>
<summary><b>Part B — Data Acquisition</b></summary>

All 4 sources loaded and merged on `customer_id`. Invalid null representations (`"None"`, `"NULL"`, `"N/A"`, `"-"`) standardized to `NaN`.

</details>

<details>
<summary><b>Part C — Data Understanding & Cleaning</b></summary>

Auto-profiling report generated via `ydata-profiling`. Missing values handled with:
- `SimpleImputer` (mean, median, most_frequent)
- `KNNImputer` ✅ selected
- `IterativeImputer` / MICE ✅ selected
- Random Sample Imputation
- Complete Case Analysis (baseline — dropped ~15% of rows)

</details>

<details>
<summary><b>Part D — Outlier Handling</b></summary>

Columns treated: `age`, `annual_income`, `loan_amount`, `credit_score`, `transaction_count`

| Method | Outcome |
|--------|---------|
| Z-Score (threshold=3) | Fewer removals |
| **IQR** ✅ | Best balance — selected |
| Percentile (5th–95th) | Over-aggressive removal |
| Winsorization | Clips extremes, zero row loss |

</details>

<details>
<summary><b>Part E — Feature Engineering (Encoding + Binning)</b></summary>

- Ordinal Encoding: `education_level` (Primary → Post-Graduate)
- Label Encoding: `gender`, `employment_type`
- One-Hot Encoding: `region`, `loan_purpose`
- Date Extraction: `year`, `month`, `day`, `weekday` from `join_date`
- Binning: Uniform, Quantile, K-Means on numerical features
- Binarization: `credit_score` → good (≥700) / poor (<700)

</details>

<details>
<summary><b>Part F — Feature Scaling</b></summary>

Compared: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`

</details>

<details>
<summary><b>Part G — Transformations & Feature Construction</b></summary>

Transformations: Log, Square Root, Reciprocal, Box-Cox, Yeo-Johnson

3 domain-engineered features:
```python
debt_to_income_ratio         = loan_amount / annual_income
average_monthly_transactions = transaction_count / months_since_join
spending_to_income_ratio     = spending_ratio / annual_income
```

</details>

<details>
<summary><b>Part H — Final Deliverable</b></summary>

`final_cleaned_data.csv` — 842 rows × 32 features, fully numeric, zero nulls, ML-ready.

</details>

### 📊 Result

| Metric | Before | After |
|--------|:------:|:-----:|
| Rows | 1,000 | **842** |
| Features | 14 | **32** |
| Missing Values | ❌ Present | **✅ Zero** |
| Outliers | ❌ Present | **✅ Removed** |
| ML Ready | ❌ | **✅** |

---

## 📊 PR 1 — Customer Churn Prediction

> Predicts customer churn using purchase behavior and demographics. Includes full EDA with 4 visualization types and a Logistic Regression classifier.

### 📁 Files

```
pr 1/
├── 📓 advance_customer_analysis.ipynb   ← Full analysis notebook
├── advanced_customer_data.csv           ← Primary dataset
├── advanced_customer_data.json          ← JSON source
├── advanced_customer_data.db            ← SQLite source
├── Part_A_Answers.pdf                   ← Conceptual theory (written answers)
├── quick_report.html                    ← ydata-profiling EDA report
├── images/
│   ├── univariate_analysis.png          ← Histogram, Box Plot, KDE
│   ├── bivariate_analysis.png           ← Box plot & Violin plot vs Churn
│   ├── multivariate_analysis.png        ← Correlation Heatmap
│   └── pairplot.png                     ← Pairplot colored by Churn
└── README.md
```

### 🎯 Objective

Predict whether a customer will **churn** (0 = retained, 1 = churned) to help businesses improve retention strategies.

### 📋 Dataset Features

| Feature | Description |
|---------|-------------|
| `CustomerID` | Unique customer identifier |
| `Age` | Customer age |
| `Gender` | Male / Female |
| `Income` | Annual income |
| `PurchaseFrequency` | Number of purchases made |
| `AvgPurchaseValue` | Average spend per purchase |
| `Tenure` | Duration of customer relationship |
| `Churn` | 🎯 Target: 0 = retained, 1 = churned |

### 🔍 Exploratory Data Analysis

**Univariate Analysis**
- Histogram: Age distribution (peak around 43)
- Box Plot: Purchase frequency spread (median ~20, range 0–40)
- KDE Plot: Income distribution (right-skewed, majority below 200K with outliers up to 600K)

**Bivariate Analysis**
- Box Plot — Income vs Churn: Similar income distributions across churn groups; outliers more extreme in non-churners
- Violin Plot — Gender vs Purchase Frequency: Nearly identical distributions for Male and Female customers

**Multivariate Analysis (Correlation Heatmap)**
- Strongest correlation: `PurchaseFrequency` ↔ `AvgPurchaseValue` (r = 0.22)
- Churn shows weak correlations with all features — non-linear model may perform better
- `Tenure` has a slight negative correlation with churn (longer tenure = less likely to churn)

**Pairplot (colored by Churn)**
- Churn class (orange) and non-churn (blue) overlap significantly across all feature pairs
- Income has the highest variance with extreme outliers in both classes

### 🤖 Machine Learning

| Setting | Value |
|---------|-------|
| Algorithm | Logistic Regression |
| Problem Type | Binary Classification |
| Target | `Churn` (0 / 1) |
| Evaluation Metric | Accuracy, F1-Score, ROC-AUC |

### 💡 Key Insights

- Customers with **low purchase frequency** are more likely to churn
- **Tenure** is negatively correlated with churn — longer relationships = better retention
- **Income outliers** are present and impact model performance
- Gender shows no significant difference in purchase behavior

---

## 🏥 PR 2 — Patient Health Risk Prediction

> Applies a focused missing value imputation and outlier removal pipeline to a healthcare dataset, producing a clean dataset for disease risk classification.

### 📁 Files

```
pr 2/
├── 📓 patient_health.ipynb              ← Preprocessing pipeline notebook
├── patient_health_dataset.csv           ← Raw dataset (1,000 patients)
├── final_clean_patient_dataset.csv      ← Cleaned output (904 patients)
└── README.md                            ← Project documentation
```

### 🎯 Objective

Predict **disease risk** (0 = low risk, 1 = high risk) for patients based on clinical and demographic indicators. Demonstrates the impact of different imputation and outlier strategies on healthcare data.

### 📋 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `patient_id` | ID | Unique patient identifier |
| `age` | Numeric | Patient age |
| `gender` | Categorical | Male / Female |
| `region` | Categorical | East / West / North / South |
| `bmi` | Numeric | Body Mass Index |
| `blood_pressure` | Numeric | Blood pressure reading |
| `cholesterol` | Numeric | Cholesterol level |
| `glucose` | Numeric | Blood glucose level |
| `disease_risk` | Binary | 🎯 Target: 0 = low risk, 1 = high risk |

### ⚙️ Pipeline

**Part A — Missing Value Imputation**

Four strategies applied and compared:

| Method | How It Works | Result |
|--------|-------------|--------|
| `SimpleImputer` | Mean for numerical, mode for categorical | Fast but ignores feature relationships |
| Random Sample Imputation | Samples from observed values; adds `MissingIndicator` | Preserves distribution |
| `KNNImputer` (k=5, distance-weighted) | Encodes categoricals → finds 5 nearest neighbors | Respects inter-feature structure |
| **`IterativeImputer` / MICE** ✅ | 250 iterations; models each feature as function of others | Most accurate — selected for downstream use |

> Categorical features (`gender`, `region`) are label-encoded before KNN/MICE and decoded back after imputation.

**Part B — Outlier Detection & Removal**

Columns treated: `age`, `bmi`, `cholesterol`, `glucose`

| Method | Rows Remaining |
|--------|---------------|
| No treatment (MICE baseline) | 1,000 |
| Z-Score (threshold = 3) | ~970 |
| **IQR** ✅ | **904** — selected |
| Percentile (5th–95th) | ~900 |

**Part C — Visual Comparison**

Side-by-side box plots comparing `glucose` distributions across all 4 outlier methods — before and after treatment.

### 📊 Result

| Metric | Raw | Cleaned |
|--------|:---:|:-------:|
| Rows | 1,000 | **904** |
| Features | 9 | **9** |
| Missing Values | ❌ Present | **✅ Zero** |
| Outliers | ❌ Present | **✅ Removed (IQR)** |
| Disease Risk (0/1) | — | **542 / 362** |
| ML Ready | ❌ | **✅** |

### 💡 Key Findings

- **MICE outperformed** all other imputation strategies — preserves complex clinical relationships between BMI, glucose, and cholesterol
- **IQR method** provided the best outlier removal — balanced data retention (904/1000 rows) without over-aggressive clipping
- Z-Score was too lenient; Percentile clipping too aggressive for medical data

---

## 🛒 Practical Exam — Customer Purchase Prediction

> **The exam project.** A full end-to-end feature engineering pipeline applied to an e-commerce dataset — ingesting from 4 source types, engineering 39 features, and producing a zero-null, ML-ready dataset to predict customer purchase behavior.

### 📁 Files

```
practical exam/
├── 📓 feature_pipeline.ipynb            ← Full pipeline notebook
├── customers.csv                         ← Customer demographics (1,000 rows)
├── transactions.json                     ← Transaction records (1,000 entries)
├── products.sql                          ← Product catalog (SQLite, 20 products)
├── processed_customer_data.csv           ← ✅ Output: 1,000 rows × 39 features
└── Summary_Report.pdf                    ← Techniques, findings & method comparison
```

### 🎯 Objective

Predict whether a customer will **make a purchase** (0 = no purchase, 1 = purchased) using engineered features derived from demographics, transaction history, and product catalog data.

```
CSV ──┐
JSON──┤──► MERGE ──► CLEAN ──► IMPUTE ──► ENGINEER ──► ENCODE ──► SCALE ──► ✅ 1,000 × 39
SQL ──┤   1,000     regex     KNN/MICE   time feats    OHE/LE    Robust
API ──┘
```

### 🔌 Data Sources

| Source | File | Key Information |
|--------|------|----------------|
| 👤 CSV | `customers.csv` | `customer_id`, `age`, `gender`, `city`, `income` |
| 💳 JSON | `transactions.json` | `transaction_id`, `customer_id`, `product_id`, `amount`, `date`, `payment_mode`, `quantity` |
| 🛍️ SQL | `products.sql` | `product_id`, `category`, `price`, `stock` (20 products across 6 categories) |
| 🌐 API | External enrichment | Supplementary customer signals |

> All datasets joined on shared keys: **`customer_id`** and **`product_id`**

### ⚙️ Pipeline Stages

<details>
<summary><b>Stage 1 — Data Ingestion & Merging</b></summary>

- Loaded all 4 sources and merged on `customer_id` and `product_id`
- Mixed-type `customer_id` values resolved using **regex-based cleaning**
- Invalid null representations standardized to `NaN`

</details>

<details>
<summary><b>Stage 2 — Missing Value Imputation</b></summary>

Missing values (~5%) in `age` and `income` — 5 strategies compared:

| Method | Rows Lost | Notes |
|--------|-----------|-------|
| Simple Imputer | 0 | Fast; ignores feature relationships |
| Random Sample | 0 | Preserves original distribution |
| **KNN Imputer** ✅ | **0** | Captures inter-feature similarity |
| **MICE** ✅ | **0** | Most accurate; computationally expensive |
| Complete Case Analysis | ~50 | Data loss — not preferred |

> 🏆 **KNN and MICE** selected — best relationship preservation, zero data loss.

</details>

<details>
<summary><b>Stage 3 — Feature Engineering</b></summary>

New features derived to boost predictive power:

| Feature | Description |
|---------|-------------|
| `days_since_signup` | Days between signup and reference date |
| `days_since_last_purchase` | Recency signal |
| `purchase_month` | Month from transaction date |
| `purchase_weekday` | Day-of-week from transaction date |
| `total_purchases` | Aggregate purchase count per customer |
| `purchase_per_day` | Purchase frequency normalized by time |
| `avg_spend_per_purchase` | Average transaction value |
| `recency_score` | Composite recency signal |
| `income_quantile_bin` | Income binned into quantile buckets |
| `frequent_buyer` | Binary flag for high-frequency customers |
| `purchased` | 🎯 **Target variable** — derived from transaction records |

</details>

<details>
<summary><b>Stage 4 — Encoding</b></summary>

- Label encoding for binary/ordinal variables (`gender`, `payment_mode`)
- One-hot encoding for nominals: `city` (16 cities), `category` (6 categories)
- Product name label-encoded

</details>

<details>
<summary><b>Stage 5 — Feature Scaling</b></summary>

| Scaler | Applied To |
|--------|------------|
| 🏆 **RobustScaler** | `income`, `transaction_amount` (skewed, outliers present) |
| **StandardScaler** | `age` (normally distributed) |

</details>

### 📊 EDA Techniques Applied

- 📊 **Histograms** — Feature distribution analysis
- 🔥 **Heatmaps** — Correlation matrix
- 🔵 **Scatter Plots** — Pairwise relationship exploration
- 🔷 **Pairplots** — Multi-dimensional pattern detection
- 📐 **Grouped Statistical Analysis** — Segment-level aggregations

### 📊 Result

| Metric | Before | After |
|--------|:------:|:-----:|
| Rows | 1,000 | **1,000** |
| Features | ~10 raw | **39 engineered** |
| Missing Values | ❌ Present | **✅ Zero** |
| ML Ready | ❌ | **✅** |

### 💡 Key Findings

- **KNN and MICE** imputation preserved relationships between `income`, `age`, and purchase behavior better than simpler methods
- **RobustScaler** was essential for `income` — heavy right skew and outliers would distort StandardScaler
- Engineered **recency and frequency features** (`days_since_last_purchase`, `frequent_buyer`) are likely to be the strongest predictors of purchase intent
- City-level one-hot encoding captured regional purchasing pattern differences across 16 Indian cities

---

## 🧠 Techniques Across All Projects

| Technique | Final PR | PR 1 | PR 2 | Practical Exam |
|-----------|:--------:|:----:|:----:|:--------------:|
| Multi-source data ingestion | ✅ (4 sources) | ✅ (3 sources) | ✅ (CSV) | ✅ (4 sources) |
| SimpleImputer | ✅ | ✅ | ✅ | ✅ |
| KNN Imputation | ✅ | ✅ | ✅ | ✅ |
| MICE / Iterative Imputation | ✅ | ✅ | ✅ | ✅ |
| Random Sample Imputation | ✅ | — | ✅ | ✅ |
| Missing Indicator | — | — | ✅ | — |
| Z-Score Outlier Removal | ✅ | — | ✅ | — |
| IQR Outlier Removal | ✅ | — | ✅ | — |
| Percentile Capping | ✅ | — | ✅ | — |
| Winsorization | ✅ | — | — | — |
| Ordinal Encoding | ✅ | — | — | — |
| Label Encoding | ✅ | — | ✅ | ✅ |
| One-Hot Encoding | ✅ | — | — | ✅ |
| Binning (Uniform/Quantile/KMeans) | ✅ | — | — | ✅ |
| Binarization | ✅ | — | — | — |
| Feature Scaling (RobustScaler) | ✅ | — | — | ✅ |
| Feature Scaling (StandardScaler) | ✅ | — | — | ✅ |
| Log / Box-Cox / Yeo-Johnson | ✅ | — | — | — |
| Domain Feature Construction | ✅ | — | — | ✅ |
| Datetime Feature Engineering | ✅ | — | — | ✅ |
| EDA (Univariate / Histograms) | ✅ | ✅ | — | ✅ |
| EDA (Bivariate / Scatter) | ✅ | ✅ | — | ✅ |
| EDA (Multivariate / Heatmap) | ✅ | ✅ | — | ✅ |
| Pairplot | — | ✅ | — | ✅ |
| Grouped Statistical Analysis | — | — | — | ✅ |
| ydata-profiling Report | ✅ | ✅ | — | — |
| Logistic Regression | — | ✅ | — | — |
| Visual Comparison (box plots) | — | — | ✅ | — |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Maitrak-2661/data-preprocessing-and-feature-engineering.git
cd data-preprocessing-and-feature-engineering
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling jupyter
```

### 3. Open any project

```bash
# Final PR — Full pipeline
cd "final pr"
jupyter notebook fixed_Holistic_data_preparer.ipynb

# PR 1 — Customer churn
cd "pr 1"
jupyter notebook advance_customer_analysis.ipynb

# PR 2 — Patient health
cd "pr 2"
jupyter notebook patient_health.ipynb

# Practical Exam — Customer purchase prediction
cd "practical exam"
jupyter notebook feature_pipeline.ipynb
```

> ⚠️ Make sure all data files are in the **same directory** as the notebook before running.

---

## 📦 Dependencies

```
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
scikit-learn >= 1.1
ydata-profiling >= 4.0
jupyter >= 1.0
```

---

## 📁 Complete Repository Structure

```
📦 data-preprocessing-and-feature-engineering/
│
├── 📁 final pr/
│   ├── fixed_Holistic_data_preparer.ipynb
│   ├── transactions.csv
│   ├── customer_metadata.json
│   ├── loan_repayment.db
│   ├── economic_indicators_api.json
│   ├── data_profiling_report.html
│   └── final_cleaned_data.csv
│
├── 📁 pr 1/
│   ├── advance_customer_analysis.ipynb
│   ├── advanced_customer_data.csv
│   ├── advanced_customer_data.json
│   ├── advanced_customer_data.db
│   ├── Part_A_Answers.pdf
│   ├── quick_report.html
│   ├── images/
│   │   ├── univariate_analysis.png
│   │   ├── bivariate_analysis.png
│   │   ├── multivariate_analysis.png
│   │   └── pairplot.png
│   └── README.md
│
├── 📁 pr 2/
│   ├── patient_health.ipynb
│   ├── patient_health_dataset.csv
│   ├── final_clean_patient_dataset.csv
│   └── README.md
│
├── 📁 practical exam/
│   ├── feature_pipeline.ipynb
│   ├── customers.csv
│   ├── transactions.json
│   ├── products.sql
│   ├── processed_customer_data.csv
│   └── Summary_Report.pdf
│
└── 📄 README.md                          ← You are here
```

---

<div align="center">

---

**Built with 💙 as part of a Data Science learning journey**

*If this helped you, consider giving it a ⭐*

</div>
