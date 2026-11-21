# Detecting Financial Fraud in Highly Imbalanced Data While Minimizing False Positives

**DAT 490 Capstone, Arizona State University**

Rahila Jafari, Yasna Kazimi, Ishita Upadhyay, Violet Guillow

## Overview

Fraudulent transactions make up only a small share of financial data, so a model can look highly accurate while missing most fraud. This project compares machine learning models for detecting fraud in highly imbalanced data, with a focus on keeping false-positive alerts low.

**Research question:** How can machine learning models effectively detect fraudulent financial transactions while minimizing false-positive alerts in highly imbalanced transaction data?

## Datasets

- [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 283,726 transactions after cleaning, 0.167% fraud
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection): 590,540 transactions, about 3.5% fraud

The datasets are not included in this repository because of their size. They can be downloaded from Kaggle using the links above.

## Notebooks

| Notebook | Contents |
|---|---|
| `Capstone_EDA.ipynb` | Data cleaning and exploratory analysis of both datasets |
| `Team1_Methodology.ipynb` | Model training, balancing-method comparison, Random Forest feature importance, and threshold analysis |

## Methods

We compared six models: Logistic Regression, Decision Tree, Random Forest, XGBoost, K-Nearest Neighbors, and Naive Bayes. Class imbalance was handled with class weighting, `scale_pos_weight`, random undersampling, and SMOTE. Because accuracy is misleading on imbalanced data, models were evaluated using precision, recall, F1-score, and the area under the precision-recall curve (AUPRC).

## Key Findings

- XGBoost and Random Forest performed best on both datasets. On ULB, XGBoost reached an AUPRC of 0.820 and Random Forest reached a precision of 0.986.
- Balancing methods mostly shifted the trade-off between precision and recall rather than improving overall performance. Random undersampling generally lowered results.
- Tuning the Random Forest threshold gave little extra benefit, since the default threshold already produced very few false alarms.

## How to Run

The notebooks were written in Google Colab. To run them, download the datasets, upload them to Google Drive, and update the file paths in the first cells to match your folder.
