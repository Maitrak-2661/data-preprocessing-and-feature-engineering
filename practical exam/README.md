<div align="center">

# 🛒 Customer Purchase Prediction Pipeline

### *End-to-End ML Feature Engineering & Data Processing*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<br/>

> **A production-grade feature engineering pipeline** that ingests raw data from CSV, JSON, SQL, and REST API sources — cleans, imputes, encodes, and scales it — and outputs a fully-processed dataset ready for ML modeling.

<br/>

![Pipeline Banner](https://img.shields.io/badge/Dataset-1%2C000%20Customers%20%7C%201%2C000%20Transactions%20%7C%2020%20Products-success?style=flat-square)
![Features](https://img.shields.io/badge/Output%20Features-39%20Engineered%20Columns-blueviolet?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

</div>

---

## 📌 Table of Contents

- [✨ Overview](#-overview)
- [📁 Repository Structure](#-repository-structure)
- [🗄️ Data Sources](#️-data-sources)
- [🔬 Pipeline Stages](#-pipeline-stages)
- [🧹 Data Quality & Imputation](#-data-quality--imputation)
- [⚖️ Feature Scaling](#️-feature-scaling)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🏗️ Feature Engineering](#️-feature-engineering)
- [📦 Output Dataset](#-output-dataset)
- [🚀 Getting Started](#-getting-started)
- [📈 Results Summary](#-results-summary)

---

## ✨ Overview

This project builds a **complete data pipeline** for predicting customer purchase behavior in an e-commerce setting. Starting from messy, multi-source raw data, the pipeline systematically transforms inputs into a clean, feature-rich dataset — ready to plug into any classification model.

```
Raw Data (CSV + JSON + SQL + API)
        ↓
  Data Merging & Cleaning
        ↓
   Missing Value Imputation
        ↓
   Feature Engineering
        ↓
  Encoding + Scaling
        ↓
processed_customer_data.csv  ✅
```

---

## 📁 Repository Structure

```
📦 customer-purchase-pipeline
 ┣ 📓 feature_pipeline.ipynb       ← Main pipeline notebook
 ┣ 📄 customers.csv                ← Raw customer demographics
 ┣ 📄 transactions.json            ← Raw transaction records (1,000 entries)
 ┣ 📄 products.sql                 ← Product catalog (SQLite schema + 20 products)
 ┣ 📄 processed_customer_data.csv  ← ✅ Final ML-ready dataset (39 features)
 ┗ 📄 Summary_Report.pdf           ← Techniques, findings & method comparison
```

---

## 🗄️ Data Sources

| Source | Format | Records | Key Fields |
|--------|--------|---------|------------|
| 👤 Customer Demographics | `CSV` | 1,000 | `customer_id`, `age`, `gender`, `city`, `income` |
| 💳 Transaction History | `JSON` | 1,000 | `transaction_id`, `customer_id`, `product_id`, `amount`, `date` |
| 🛍️ Product Catalog | `SQL` | 20 | `product_id`, `category`, `price`, `stock` |
| 🌐 External Enrichment | `REST API` | — | Supplementary customer signals |

> All datasets joined on shared keys: **`customer_id`** and **`product_id`**

---

## 🔬 Pipeline Stages

### Stage 1 — 🔗 Data Ingestion & Merging
- Loaded data from 4 heterogeneous sources
- Unified on common foreign keys
- Resolved mixed-type `customer_id` values using **regex-based cleaning**

### Stage 2 — 🧹 Cleaning
- Detected & treated missing values in `age` (~5%) and `income` (~5%)
- Standardized data types across all columns
- Validated referential integrity between datasets

### Stage 3 — 🏗️ Feature Engineering *(see below)*

### Stage 4 — 🔢 Encoding
- Label encoding for ordinal variables
- One-hot encoding for nominal variables (`city`, `category`)
- Binary target variable `purchased` derived from transaction records

### Stage 5 — ⚖️ Scaling
- Applied optimal scalers per feature distribution

---

## 🧹 Data Quality & Imputation

Missing values were handled using **5 competing strategies** — evaluated for accuracy vs. data preservation:

| Method | Rows Lost | Notes |
|--------|-----------|-------|
| 🟡 Simple Imputer | 0 | Fast; ignores feature relationships |
| 🟡 Random Sample | 0 | Preserves original distribution |
| 🟢 **KNN Imputer** | **0** | **Captures inter-feature similarity** ✅ |
| 🟢 **MICE** | **0** | **Most accurate; computationally expensive** ✅ |
| 🔴 Complete Case Analysis | ~50 | Data loss — not recommended |

> 🏆 **Winner:** KNN and MICE imputation — best relationship preservation, zero data loss.

---

## ⚖️ Feature Scaling

| Scaler | Best For | Applied To |
|--------|----------|------------|
| 🏆 **RobustScaler** | Skewed data with outliers | `income`, `transaction_amount` |
| **StandardScaler** | Normally distributed features | `age` |

---

## 📊 Exploratory Data Analysis

The following EDA techniques were applied to understand distributions, correlations, and patterns:

- 📊 **Histograms** — Feature distribution analysis
- 🔥 **Heatmaps** — Correlation matrix across all numeric features
- 🔵 **Scatter Plots** — Pairwise relationship exploration
- 🔷 **Pairplots** — Multi-dimensional pattern detection
- 📐 **Grouped Statistical Analysis** — Segment-level aggregations

---

## 🏗️ Feature Engineering

New features derived to boost predictive power:

| Feature | Description |
|---------|-------------|
| `days_since_signup` | Days between customer signup and reference date |
| `days_since_last_purchase` | Recency signal for purchase behavior |
| `purchase_month` | Month extracted from transaction date |
| `purchase_weekday` | Day-of-week extracted from transaction date |
| `total_purchases` | Aggregate purchase count per customer |
| `purchase_per_day` | Purchase frequency normalized by time |
| `avg_spend_per_purchase` | Average transaction value per customer |
| `recency_score` | Composite recency signal |
| `income_quantile_bin` | Income binned into quantile buckets |
| `frequent_buyer` | Binary flag for high-frequency customers |
| `purchased` | 🎯 **Target variable** — derived from transaction records |

---

## 📦 Output Dataset

**File:** `processed_customer_data.csv`
**Shape:** `1,000 rows × 39 features`

Includes engineered time features, encoded categoricals, scaled numerics, and the binary `purchased` target — ready for direct use with any sklearn-compatible model.

```python
import pandas as pd

df = pd.read_csv("processed_customer_data.csv")
X = df.drop("purchased", axis=1)
y = df["purchased"]
# → Feed directly into your classifier 🚀
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the Pipeline

```bash
# Clone the repo
git clone https://github.com/your-username/customer-purchase-pipeline.git
cd customer-purchase-pipeline

# Launch the notebook
jupyter notebook feature_pipeline.ipynb
```

### Quick Data Load

```python
import pandas as pd
import json
import sqlite3

# Load customers
customers = pd.read_csv("customers.csv")

# Load transactions
with open("transactions.json") as f:
    transactions = pd.DataFrame(json.load(f))

# Load products from SQL
conn = sqlite3.connect(":memory:")
with open("products.sql") as f:
    conn.executescript(f.read())
products = pd.read_sql("SELECT * FROM products", conn)
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| 📥 Raw input sources | 4 (CSV, JSON, SQL, API) |
| 👥 Customers | 1,000 |
| 💳 Transactions | 1,000 |
| 🛍️ Products | 20 |
| 🔧 Engineered features | 39 |
| ❌ Missing values remaining | 0 |
| 🎯 Target variable | `purchased` (binary) |

---

<div align="center">

**Built with ❤️ for clean data and better predictions**

⭐ *Star this repo if you found it useful!* ⭐

</div>
