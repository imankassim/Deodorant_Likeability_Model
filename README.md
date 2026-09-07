# Logistic regression model to determine deodorant likeability

This is a logistic regression model used to determine deodorant likeability. It's a learning project exploring data cleaning, feature engineering, and classification techniques with `scikit-learn` and `statsmodels`.

Specifically, it predicts which of two deodorants — **Deodorant B** (Sure Invisible Antiperspirant) and **Deodorant J** (Lynx Africa) — a person prefers, based on survey data covering demographics (ethnicity, education, income, relationship status) and product perception ratings (e.g. attractive, cheap, long lasting, overpowering).

## What the notebook does

1. **Data source**: Loads survey response data from a public S3 CSV.
2. **Filtering/cleanup**: Drops irrelevant columns and narrows the dataset to the two products being compared.
3. **Feature engineering**: Averages groups of perception questions into `pos_average` / `neg_average` scores, then drops the raw question columns.
4. **Exploration**: Bar plots and cross-tabs of product preference by ethnicity, education, income, and relationship status.
5. **Encoding**: One-hot encodes categorical variables into dummy columns; encodes the product choice as 0/1.
6. **Balancing**: Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the classes.
7. **Feature selection**: Uses Recursive Feature Elimination (RFE) to select the most predictive features.
8. **Modeling**: Fits a `statsmodels` Logit model for a statistical summary, then trains an `sklearn` `LogisticRegression` classifier on the selected features.
9. **Evaluation**: Reports a confusion matrix, precision/recall/F1 scores, and an ROC curve.

## Getting Started

The model can be run through Google Colab, as linked.

### Prerequisites

None
