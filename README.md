# B2B SaaS Customer Churn Prediction

## Project Overview

Customer churn can reduce recurring revenue and make growth less predictable for a business-to-business software company. This project explores whether customer, subscription, usage, support, and billing information can be used to identify accounts labeled as churned.

The project uses a supervised machine-learning workflow in a Jupyter notebook. It loads and explores a 50,000-row dataset, prepares the data, trains a Logistic Regression classifier, and evaluates its predictions on a held-out test set.

The notebook provides an end-to-end learning and analysis workflow. It is not currently a deployed application or a production-ready churn service.

## Objective

Build a binary classifier for the `Is_Churned` target:

| Value | Meaning |
| --- | --- |
| `0` | Stay |
| `1` | Churn |

The model's predictions may help prioritize customer-retention analysis. They should be treated as signals for further review, not as definitive explanations of why a customer churns.

## Dataset

The project starts with `MP_Synthetics_50000_Rows.csv`, which contains 50,000 customer records. The target distribution found during exploration is:

| Target | Count |
| --- | ---: |
| Stay (`0`) | 49,084 |
| Churn (`1`) | 916 |

The target is therefore imbalanced: churn accounts for a small fraction of the records. This is important when interpreting accuracy and when choosing evaluation measures.

The input includes fields describing:

- Customer and company context, such as industry, country, and company size
- Subscription details, such as plan tier, billing cycle, and monthly recurring revenue
- Account management and payment status
- Signup and last-active dates
- Seat usage, recent support tickets, and feature adoption
- Churn risk score
- The churn label, `Is_Churned`

The notebook removes identifier and company-name fields (`Customer_ID` and `Company_Name`) from the modeling data. The dataset is included locally in this project folder.

## End-to-End Workflow

The workflow is implemented in `project 1 supervised learning.ipynb`.

### 1. Load and explore the data

- Read `MP_Synthetics_50000_Rows.csv` into a pandas DataFrame.
- Inspect the dataset's dimensions, columns, data types, missing values, and duplicate rows.
- Review categorical values and the distribution of `Is_Churned`.
- Identify data quality issues, including repeated column names.

### 2. Clean and prepare the data

- Remove `Customer_ID` and `Company_Name`, which are not used as model predictors.
- Remove duplicate records and repeated column labels, keeping the first occurrence of a repeated label.
- Trim whitespace and standardize selected categorical values.
- Parse `Signup_Date` and `Last_Active_Date` as dates.
- Convert `Feature_Adoption_Rate` and `Churn_Risk_Score` to numeric values, treating invalid values as missing.
- Save the prepared dataset as `cleaned_churn_data.csv`.

### 3. Define features and target

- Use the remaining columns other than `Is_Churned` as the predictors (`X`).
- Use `Is_Churned` as the target (`y`). It is already encoded as `0` and `1`, so it does not need a `"No"`/`"Yes"` mapping.
- One-hot encode categorical predictors with `pandas.get_dummies(..., drop_first=True)`.

### 4. Split the data

- Split the observations into 80% training data and 20% test data.
- Use `random_state=42` so the split is reproducible.

### 5. Impute and train

- Remove predictor columns that are entirely missing in the training data.
- Fill remaining missing predictor values using medians calculated from the training set, and apply those same medians to the test set.
- Train a scikit-learn Logistic Regression classifier with `max_iter=1000`.

### 6. Predict and evaluate

- Generate predictions for the held-out test data.
- Report accuracy, precision, recall, and F1-score by class.
- Display a confusion matrix in class order `[0, 1]` (Stay, then Churn).

## Data Flow

```text
MP_Synthetics_50000_Rows.csv
              |
              v
   Exploration and data cleaning
              |
              +------> cleaned_churn_data.csv
              |
              v
 Feature/target separation and encoding
              |
              v
      80/20 train-test split
              |
              v
 Missing-value imputation and Logistic Regression
              |
              v
     Predictions and evaluation
```

## Model

The notebook uses **Logistic Regression**, a supervised classification algorithm suitable for a two-class prediction task. The model estimates whether a record belongs to the Stay (`0`) or Churn (`1`) class from the prepared predictor columns.

## Current Test Results

The notebook's current 10,000-record test set contains 9,823 Stay cases and 177 Churn cases. Its confusion matrix is:

| Actual / Predicted | Stay (`0`) | Churn (`1`) |
| --- | ---: | ---: |
| Stay (`0`) | 9,823 | 0 |
| Churn (`1`) | 7 | 170 |

This corresponds to:

- **Accuracy:** 99.93% (`0.9993`)
- **Churn precision:** 100% at the displayed precision
- **Churn recall:** approximately 96.05% (`170` of `177` churn cases detected)
- **Churn F1-score:** approximately 0.98
- **False negatives:** 7 churn cases predicted as Stay
- **False positives:** 0 Stay cases predicted as Churn

These are the notebook's current results, not a guarantee of performance on new customers. Because the data is imbalanced and the results are unusually strong, they should be independently validated before any business use.

## Evaluation Metrics

| Metric | What it measures |
| --- | --- |
| Accuracy | Fraction of all test predictions that are correct |
| Precision | Fraction of predicted churn cases that are actually churn |
| Recall | Fraction of actual churn cases that the model identifies |
| F1-score | Harmonic balance of precision and recall |
| Confusion matrix | Counts of correct and incorrect predictions for each class |

For a churn use case, churn recall and precision are particularly useful alongside accuracy: missing a likely churn account and incorrectly flagging a staying account can have different business costs.

## Technologies

- Python
- Jupyter Notebook
- pandas
- scikit-learn

## Project Structure

```text
B2B Saas Churn dataset/
├── MP_Synthetics_50000_Rows.csv
├── cleaned_churn_data.csv
├── project 1 supervised learning.ipynb
└── README.md
```

`cleaned_churn_data.csv` is the prepared output written by the notebook. Rerunning the cleaning workflow updates that output.

## Run the Project

1. Open a terminal in this project folder.
2. Install the required packages if they are not already available:

   ```bash
   python -m pip install pandas scikit-learn jupyter
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `project 1 supervised learning.ipynb` and run the cells from top to bottom. Keep `MP_Synthetics_50000_Rows.csv` in the same folder so the notebook can load it.

## Business Use Case

A SaaS team could use a validated churn model to help prioritize accounts for review and retention outreach. For example, churn-risk predictions could inform customer-success follow-up, account-health checks, or further analysis of service and product usage.

Predictions alone do not identify the cause of churn or prove that a particular retention action will work. Any outreach strategy should be evaluated for effectiveness and reviewed in the context of customer needs.

## Limitations and Next Steps

- **Check for target leakage:** Determine how `Churn_Risk_Score` was created and whether it uses `Is_Churned` or information recorded after a customer churned. Exclude it if it would not be available at the time a real prediction is made.
- **Use a stratified split:** Set `stratify=y` when splitting so the rare churn class is represented consistently in training and test data.
- **Establish baselines:** Compare the model with a simple majority-class baseline and evaluate whether it adds useful churn detection.
- **Validate more robustly:** Use cross-validation and a separate, appropriately sampled test set.
- **Assess threshold and imbalance strategies:** Examine precision-recall trade-offs and consider class weighting or other imbalance-handling methods.
- **Review features and timing:** Ensure every predictor is available before the prediction point; engineer date features in a way that avoids using future information.
- **Extend the evaluation:** Add ROC-AUC and precision-recall analysis, and compare Logistic Regression with other suitable classifiers.
- **Prepare for deployment only after validation:** A real service would need a repeatable preprocessing-and-model pipeline, monitoring, and a process for retraining and reviewing outcomes.

Author- Viswaraja
