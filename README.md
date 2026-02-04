# Preterm Birth Prediction Using Machine Learning

A machine learning system for predicting preterm births using uterine electromyography (EMG) signals. This project demonstrates end-to-end ML workflow from exploratory data analysis to model deployment.

## Problem

Preterm birth (delivery before 37 weeks of gestation) affects 10-15% of pregnancies worldwide and is the leading cause of neonatal mortality. Early identification of at-risk pregnancies enables timely medical intervention, potentially reducing complications and improving outcomes.

This project develops a non-invasive prediction system using uterine EMG signal features to classify pregnancies as preterm or term.

## Dataset

- **Source:** Synthetic uterine EMG recordings
- **Size:** 1,000 samples
- **Features:** 9 engineered features from EMG signals
  - Frequency domain: Mean/Median Frequency, Peak Frequency, Power bands
  - Time domain: RMS, Signal Energy
  - Nonlinear: Sample Entropy, Nonlinear Feature
- **Target:** Binary classification (0 = Term, 1 = Preterm)
- **Class distribution:** 80% term births (800), 20% preterm births (200)

## Methodology

### 1. Exploratory Data Analysis
- Statistical profiling of features
- Missing value analysis (~1-2% per feature)
- Class distribution visualization
- Feature correlation analysis

### 2. Data Preprocessing
- Missing value imputation (median strategy)
- Feature scaling (StandardScaler)
- 80-20 stratified train-test split

### 3. Model Development & Comparison
Three algorithms evaluated:
- **Random Forest** - Ensemble method, robust to noise
- **Gradient Boosting** - Sequential error correction
- **K-Nearest Neighbors** - Instance-based baseline

Hyperparameter optimization via grid search with 5-fold cross-validation.

## Results

| Model | Sensitivity | Specificity | AUC |
|-------|-------------|-------------|-----|
| **Random Forest** | **0.99** | **0.95** | **1.00** |
| Gradient Boosting | 0.94 | 0.93 | 0.99 |
| KNN | 0.88 | 0.85 | 0.94 |

### Selected Model: Random Forest

**Why Random Forest?**
- **99% sensitivity:** Detects 213 out of 216 preterm cases (only 3 missed)
- **95% specificity:** Minimizes false alarms (10 false positives)
- **Perfect AUC (1.00):** Complete class separation
- **Clinical relevance:** Missing a preterm birth has severe consequences; high sensitivity is critical

**Confusion Matrix:**
```
[[206  10]
 [  3 213]]
```

## Repository Structure

```
preterm/
├── Model.ipynb                        # Main analysis notebook
├── preterm_birth_dataset.csv          # EMG feature dataset
├── random_forest_model.pkl            # Trained classifier
├── preterm_birth_dataset_cleaned.csv  # Cleaned Dataset
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── .gitignore                         # Git exclusions
```

## Installation & Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Setup
```bash
# Clone repository
git clone https://github.com/amirosseinz/preterm-birth-emg-ml.git
cd preterm

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook Model.ipynb
```

### Using the Trained Model
```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example input (9 EMG features)
features = [0.308719, 0.372998, 0.353886, 0.097027, 
            0.596872, 0.864639, 0.313924, 6.832061, 1.222627]

# Preprocess and predict
scaled_features = scaler.transform(np.array(features).reshape(1, -1))
prediction = model.predict(scaled_features)[0]
probability = model.predict_proba(scaled_features)[0, 1]

print(f"Prediction: {'Preterm' if prediction == 1 else 'Term'}")
print(f"Preterm probability: {probability:.2%}")
```

## Key Technical Decisions

1. **Sensitivity over accuracy:** In medical applications, missing a preterm case (false negative) is more costly than a false alarm (false positive)
2. **Random Forest over Gradient Boosting:** Marginal AUC improvement (1.00 vs 0.99) with better sensitivity and lower computational cost
3. **Median imputation:** More robust to outliers than mean; preserves sample size vs. deletion
4. **Stratified split:** Maintains class balance in train/test sets

## Limitations & Future Work

**Current Limitations:**
- Synthetic dataset (requires validation on clinical data)
- Limited to 9 engineered features (raw signal analysis may capture more patterns)
- Single modality (EMG only; could integrate maternal demographics)

**Potential Enhancements:**
- Validate on real hospital EMG recordings
- Deep learning models (CNN/LSTM) for raw signal processing
- SHAP analysis for model explainability
- REST API for clinical integration
- Model monitoring for production deployment

## Technologies Used

- **Python 3.8**
- **scikit-learn** - Machine learning models
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Visualization
- **Jupyter** - Interactive development



This project is for educational and portfolio purposes.

## Contact

For questions or collaboration: [Your contact information]
