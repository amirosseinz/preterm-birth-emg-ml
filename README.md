# Synthetic EMG classification

A reproducibility study on 1,000 synthetic feature rows representing uterine EMG measurements. The data contain no patient observations. Results describe performance on this synthetic dataset.

## Evaluation correction

The original notebook upsampled the positive class by sampling rows with replacement, then split the expanded dataset. Re-running its split finds **208 of 432 test rows with identical features in training**. That invalidates the notebook's original holdout estimate. The cleaned CSV and notebook remain available as historical inputs.

The supported evaluation splits the 1,000 raw rows first, keeps the original prevalence, and fits imputation and scaling on training data through scikit-learn pipelines. It rejects exact duplicate feature rows before splitting. The fixed holdout contains 200 rows, including 40 positive examples.

| Model | Balanced accuracy | ROC AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Majority class | 0.500 | 0.500 | 0.000 | 1.000 |
| Logistic regression | 0.781 | 0.895 | 0.575 | 0.988 |
| Random forest | 0.806 | 0.896 | 0.625 | 0.988 |

The random forest identified 25 of the 40 positive test examples and missed 15. The experiment uses a single synthetic data source and one holdout. Generalisation to real measurements is unknown.

## Reproduce

```bash
python -m pip install -r requirements-evaluation.txt
python evaluate.py --audit-legacy preterm_birth_dataset_cleaned.csv
```

[Recorded results and input hash](evaluation.json) · [Original exploratory notebook](Model.ipynb)
