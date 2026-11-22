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

## Uncertainty on the fixed synthetic holdout

The evaluation reports 95% Wilson score intervals with the positive and negative test counts for each model.

| Model | Sensitivity with 95% interval | Specificity with 95% interval |
|---|---:|---:|
| Majority class | 0.000 [0.000, 0.088] | 1.000 [0.977, 1.000] |
| Logistic regression | 0.575 [0.422, 0.715] | 0.988 [0.956, 0.997] |
| Random forest | 0.625 [0.470, 0.758] | 0.988 [0.956, 0.997] |

The random forest minus logistic regression ROC AUC difference is **0.0010**, with a 95% paired bootstrap interval of **[-0.0614, 0.0527]**. This comparison does not establish an AUC advantage for random forest on the fixed holdout.

The comparison resamples test rows jointly for both models, separately within the 40 positive and 160 negative rows. It uses 2,000 replicates, seed 42 and the 2.5th and 97.5th percentiles. These intervals assume independent test rows and condition on the fitted models and observed class counts. They exclude variation from retraining, different train/test splits, the synthetic data generation process and real-world distribution shifts. The original point metrics and split remain the same.

[Wilson interval reference](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm) · [Paired bootstrap reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

## Reproduce

```bash
python -m pip install -r requirements-evaluation.txt
python evaluate.py --audit-legacy preterm_birth_dataset_cleaned.csv
python -m unittest test_evaluate -v
```

[Recorded results and input hash](evaluation.json) · [Original exploratory notebook](Model.ipynb)
