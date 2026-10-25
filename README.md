# Credit Scoring Model

Predicts whether a customer will experience serious delinquency (90+ days past due) within two years, using the "Give Me Some Credit" dataset. Built for the CodeAlpha Machine Learning internship (Task 1).

## Dataset
Kaggle competition: *Give Me Some Credit* (`cs-training.csv`, 150,000 customers, 11 columns).
The data is not included in this repo. Download it from Kaggle and place it next to the notebook.
Only about 6.7% of customers default, so the classes are heavily imbalanced.

## Approach
1. **Data cleaning**
   - A first copy of the file was corrupted (numbers like `1.046.279.103` in two columns), so I switched to a clean copy.
   - Removed 609 duplicate rows, 1 customer with age 0, 225 rows using the special codes 96/98 in the late-payment columns, and 241 rows with credit utilization above 10.
   - Final size: 148,924 rows.
2. **Missing values**
   - `MonthlyIncome` was missing for ~20% of customers. For those rows `DebtRatio` was also unreliable (implausibly huge values), so it was set to missing as well.
   - Added an `IncomeMissing` flag. Remaining gaps were filled with medians.
3. **Feature engineering:** `TotalLatePayments`, `HasLatePayment`, `IncomePerPerson`, `IncomeMissing`.
4. **Split and leakage control:** 80/20 stratified train/test split. Medians and outlier caps (99th percentile) were computed on the training set only and then applied to both sets.
5. **Models:** Logistic Regression (scaled), Decision Tree, Random Forest, plus a dummy baseline that always predicts "no default". Class weights were balanced to account for the imbalance.

## Results (test set, 29,785 customers)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Dummy (always 0) | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.213 | 0.762 | 0.333 | 0.857 |
| Decision Tree | 0.192 | 0.802 | 0.310 | 0.855 |
| Random Forest | 0.231 | 0.749 | 0.353 | 0.863 |

- Accuracy is misleading here: the dummy model scores 93% accuracy while catching no defaulters.
- Random Forest is slightly ahead, but the three models are close, and I used a single split, so the gap may not be meaningful.
- Random Forest confusion matrix: 1,476 defaulters caught, 495 missed, 4,914 good customers wrongly flagged. Balancing the classes trades precision for recall, which suits lending, where a missed defaulter usually costs more than a false alarm.
- Most important features: credit utilization, then late-payment features (`TotalLatePayments`, `HasLatePayment`). The `IncomeMissing` flag added almost nothing.

## Limitations
- One train/test split and no cross-validation or hyperparameter tuning.
- The dataset is a third-party re-upload of the Kaggle file.
- Treating `DebtRatio` as missing wherever income is missing was a judgment call.

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn (Google Colab)
