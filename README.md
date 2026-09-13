# 🩸 Blood Donation Prediction

A machine learning project that predicts whether a blood donor will donate again in a target month, based on their past donation history.

## Problem Statement

Blood banks need to anticipate which past donors are likely to return, so outreach and scheduling can be targeted efficiently. Using historical donation records, this project builds a binary classification model to predict whether a donor **made a donation in March 2007**, based on:

- Months since their last donation
- Total number of past donations
- Months since their first donation
- Total volume donated

## Dataset

The dataset (`data/donations.csv`) contains 576 donor records with the following columns:

| Column | Description |
|---|---|
| `Months since Last Donation` | Months elapsed since the donor's most recent donation |
| `Number of Donations` | Total number of times the donor has donated |
| `Total Volume Donated (c.c.)` | Total blood volume donated, in cubic centimeters |
| `Months since First Donation` | Months elapsed since the donor's first-ever donation |
| `Made Donation in March 2007` | Target variable — 1 if the donor donated that month, 0 otherwise |

## Approach

1. **EDA** — univariate, bivariate, and multivariate analysis to understand donation patterns and class balance (via `ydata-profiling`)
2. **Preprocessing** — duplicate removal, outlier handling via winsorization, skewness correction (log and square-root transforms), Min-Max scaling
3. **Class imbalance** — SMOTE was tested but reduced accuracy, so the original class distribution was kept
4. **Modeling** — eight classifiers were trained and hyperparameter-tuned:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - Support Vector Classifier (SVC)
   - K-Nearest Neighbors
   - Artificial Neural Network (MLP)
5. **Model comparison** — accuracy compared across all models to select the best performer

## Results

The **Support Vector Classifier** delivered the strongest and most consistent performance among all models tested, followed closely by the Decision Tree Classifier — making SVC the recommended model for this dataset.

## Project Structure

```
blood-donation-prediction/
├── data/
│   └── donations.csv
├── notebooks/
│   └── blood_donation_prediction.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/ABINAYALAKSHMI21/blood-donation-prediction.git
cd blood-donation-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/blood_donation_prediction.ipynb
```

## Challenges Faced

- Extracting meaningful insights from a small, imbalanced dataset
- Identifying and handling duplicate records
- Detecting and treating outliers without distorting the distribution
- Correcting skewness across multiple numerical features
- Longer training time for ensemble and kernel-based models
- Finding a random state where all models performed consistently well

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `ydata-profiling`

## License

This project is available under the MIT License.
