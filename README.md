# Customer Segmentation Project

Customer segmentation using K-Means clustering on behavioral and demographic data — identifies distinct customer groups and provides data-driven marketing recommendations.

## Overview

This project segments a base of 600 customers using unsupervised machine learning (K-Means clustering) based on their purchasing behavior and demographics. The goal is to uncover distinct customer groups that can inform targeted marketing, retention strategy, and personalized customer experiences.

## Features

- **Data Generation** — a realistic sample dataset combining demographic (age, gender, region) and behavioral (purchase frequency, total spend, recency, tenure) attributes
- **Feature Scaling** — standardization of numeric features so no single variable dominates the clustering
- **Optimal K Selection** — Elbow Method (inertia) and Silhouette Score used together to choose the right number of clusters
- **K-Means Clustering** — segments customers into 4 distinct, business-interpretable groups
- **PCA Visualization** — reduces the feature space to 2D to visually confirm cluster separation
- **Segment Profiling** — average behavior/demographics per segment, plus preferred product category breakdown
- **Business Recommendations** — actionable marketing strategy per segment

## Segments Identified

| Segment | % of Customers | Key Traits | Recommended Action |
|---|---|---|---|
| **High-Value Loyal** | 23.8% | High income, frequent purchases, high spend, very recent activity | Reward with loyalty tiers & exclusive perks |
| **Occasional Low-Spend** | 33.2% | Young, low income, infrequent low-value purchases, short tenure | Onboarding campaigns & first-purchase incentives |
| **Moderate Regular** | 27.3% | Middle-aged, steady moderate spend and frequency | Cross-sell & bundle promotions |
| **At-Risk High-Value** | 15.7% | High historical spend, longest tenure, but inactive for ~6 months | Urgent win-back / re-engagement campaigns |

## File Structure

| File | Description |
|---|---|
| `Customer_Segmentation_Report.docx` | Full write-up: objective, methodology, findings, and recommendations |
| `customer_data.csv` | Raw sample dataset (600 customers, demographic + behavioral features) |
| `customer_segments.csv` | Same data with cluster assignment and segment label added |
| `segment_profiles.csv` | Aggregated average metrics per segment |
| `generate_data.py` | Generates the sample customer dataset |
| `cluster_analysis.py` | Scales features, selects optimal K, runs K-Means, applies PCA |
| `visualize.py` | Produces all charts (elbow/silhouette, PCA scatter, segment sizes, profile comparisons, category preference) |
| `charts/` | All generated visualization images |

## How to Reproduce

```bash
pip install pandas numpy scikit-learn matplotlib

python generate_data.py      # creates customer_data.csv
python cluster_analysis.py   # runs clustering, creates customer_segments.csv & segment_profiles.csv
python visualize.py          # generates all charts in the working directory
```

To use your own data, replace `customer_data.csv` with your dataset (keeping the same column names) and rerun `cluster_analysis.py` and `visualize.py`.

## Tech Stack

- **pandas / NumPy** — data manipulation
- **scikit-learn** — StandardScaler, KMeans, PCA, silhouette_score
- **Matplotlib** — visualization

## Learning Outcomes

This project demonstrates:
- End-to-end customer analytics workflow, from raw data to business recommendations
- Unsupervised machine learning (K-Means clustering) and cluster validation techniques
- Dimensionality reduction (PCA) for visualizing high-dimensional clusters
- Translating data science output into targeted, actionable marketing insights
