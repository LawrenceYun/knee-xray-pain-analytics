# Knee X-ray Pain Analytics and Fairness Modeling

## Overview

This project investigates how structural knee X-ray features and demographic factors explain patient-reported knee pain severity in the Osteoarthritis Initiative (OAI) baseline cohort.

The analysis combines semi-quantitative radiographic scores, KOOS/WOMAC pain survey responses, and demographic variables to build interpretable machine learning models for pain prediction. The project focuses not only on predictive performance, but also on explainability and fairness across racial groups.

The main goal is to understand:

- Which X-ray structural features are associated with pain severity
- Whether demographic factors improve pain prediction beyond radiographic findings
- Which features are sufficient or necessary for individual-level model predictions
- Whether model residuals reveal systematic racial pain-reporting disparities

## Dataset

The project uses the Osteoarthritis Initiative (OAI) baseline visit (V00) dataset.

### Data Sources

The analysis merges three OAI data tables:

| File | Content |
|---|---|
| `oai_kxrsemiquant01.csv` | Semi-quantitative knee X-ray structural scores |
| `oai_koos_womac01.csv` | KOOS/WOMAC pain survey items |
| `oai_enrollee01.csv` | Demographic variables including race, sex, age, BMI, and site |

### Dataset Scale

- Cohort: OAI baseline visit, V00
- Participants: 4,000+ participants
- Primary modeling sample: 6,705 baseline right knees
- Structural predictors: 32 semi-quantitative X-ray features
- Full feature set: 37 predictors, including 32 radiographic features and 5 demographic variables
- Outcome: continuous KOOS/WOMAC pain score

## Outcome Definition

The target variable is a continuous KOOS/WOMAC pain score derived from patient-reported pain survey items.

Higher scores indicate worse pain severity.

The project uses a KOOS-first / WOMAC-fallback construction strategy to maximize usable pain outcome coverage.

## Methods

### 1. Data Processing

The data preparation pipeline includes:

- Inner join across OAI radiographic, pain survey, and demographic tables
- Subject-, visit-, and date-level matching
- Right-knee specific filtering for comparison with prior literature
- Duplicate-column resolution
- Median imputation for missing values
- Construction of a continuous KOOS/WOMAC pain score
- Separation of structural and demographic predictors

### 2. Predictive Modeling

Two XGBoost regression models are compared:

| Model | Features |
|---|---|
| Model-S | 32 semi-quantitative X-ray structural features |
| Model-SF | Structural features + demographic variables |

The objective is to measure whether demographic variables explain additional variance in patient-reported pain beyond structural X-ray findings.

### 3. Explainability

The project applies two explainability methods:

#### TreeSHAP

TreeSHAP is used to compute global feature importance for the XGBoost model with both structural and demographic features.

Top features include:

- Race
- X-ray osteophyte features
- Site
- Age
- Kellgren–Lawrence grade
- Sex

#### SANE: Sufficient and Necessary Explanations

SANE is used to identify minimal feature sets that are sufficient or necessary for individual-level model predictions.

This project implements a brute-force SANE search over selected feature subsets to improve determinism and interpretability.

### 4. Fairness Analysis

Because pain is modeled as a continuous outcome, standard Equalized Odds classification metrics are not directly suitable. Instead, the project uses continuous fairness diagnostics:

- Residual gap analysis by race
- Pain score distribution comparison by race
- Pain-versus-structural-severity comparison
- Propensity-score matching between Black and White participants based on X-ray severity

## Key Results

### Predictive Performance

Adding demographic variables improved explained variance:

| Model | R² |
|---|---:|
| Structural-only model | 0.042 |
| Structural + demographic model | 0.092 |

This suggests that demographic variables explain additional variation in patient-reported pain beyond radiographic structural features.

### Explainability Findings

TreeSHAP identified race, site, age, and several X-ray structural features as important predictors of pain score.

SANE analysis showed that race was often sufficient but not necessary, suggesting that it may act as a proxy when structural cues are weak, while richer X-ray information can partially replace it.

### Fairness Findings

The project found systematic racial residual differences after accounting for X-ray structural features.

In the residual analysis, Black participants showed lower reported pain scores relative to structural severity compared with White participants. Propensity-score matching further supported this pattern after matching on X-ray severity.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP / TreeSHAP
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

## How to Run
pip install -r requirements.txt
1. Place authorized OAI data files in the data/ directory.
2. Run the data preparation notebook or script.
3. Train structural-only and structural-demographic XGBoost models.
4. Generate SHAP feature attribution plots.
5. Run SANE sufficiency/necessity analysis.
6. Evaluate residual fairness gaps across racial groups.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── reports/
│   └── X-ray Knee Zhang.pdf
└── figures/
    └── README.md
