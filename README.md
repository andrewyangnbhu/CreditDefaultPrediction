# CreditDefaultPrediction
Repository for Predicting Credit Card Default in Young Adults: A Machine Learning Approach Using Non-Transactional Survey Data
Overview
This project predicts credit card default risk among young adults (18-34) using non-transactional behavioral and demographic data from the 2021 National Financial Capability Study (NFCS). Unlike traditional approaches that rely on credit history and transaction patterns, this model identifies at-risk individuals before warning signs appear in their account behavior.
Key Findings

86.5% ROC-AUC achieved using XGBoost
51.5% Precision at 65% recall (vs. 19.3% baseline)
Payday loan usage identified as the strongest predictor of default risk
Behavioral factors outperform purely demographic models

Table of Contents

Problem Statement
Dataset
Methodology
Installation
Usage
Results
Model Comparison
Limitations
Future Work
Citation

Problem Statement
Can we predict credit card default risk using only survey-based behavioral and demographic data, before transactional warning signs emerge? This research addresses the critical need for early intervention in financial distress, enabling proactive rather than reactive risk management.
Dataset
Source: 2021 National Financial Capability Study (NFCS)

Sample Size: 7,705 young adults (ages 18-34)
Features: 30 non-transactional predictors
Target: Credit card late fees (binary outcome)
Class Distribution: 19.3% positive class (late fees)

Excluded Features
The following transactional/outcome variables were intentionally excluded to ensure the model can predict before problems occur:

Payment behavior (paid in full, carried balance, minimum payment)
Over-limit fees and cash advances
Late student loan payments
Debt collection contacts

Top Predictors

G25_2 - Payday loan usage (past 5 years)
F1 - Number of credit cards owned
J32 - Spending vs. income comparison
J10 - Difficulty covering monthly expenses
B4 - Financial satisfaction

Methodology
1. Data Preprocessing
python# Handle missing values (98/99 codes)
# Label encode categorical variables
# Impute using median (numerical) and mode (categorical)
# Rare category consolidation (<5% threshold)
2. Feature Selection

XGBoost-based feature importance (gain metric)
Selected top 30 features from 100+ candidates
Validated using cross-validation

3. Model Development
Three approaches compared:

XGBoost Classifier (primary model)
Random Forest Classifier
Logistic Regression

4. Hyperparameter Optimization

Optuna framework (100 trials)
Optimized for precision while maintaining recall ≥65%
Custom objective function balancing false positives and false negatives

5. Threshold Tuning

Decision threshold optimized for precision
Target: Precision >50% at Recall ≥65%
Final threshold: Custom-tuned per model

Installation
Prerequisites
bashPython 3.8+
pip
Dependencies
bashpip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn
Clone Repository
bashgit clone https://github.com/yourusername/credit-default-prediction.git
cd credit-default-prediction
Usage
Quick Start
python# Load the trained model
import pickle
with open('best_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict_proba(X_test)[:, 1]
predicted_classes = (predictions >= optimal_threshold).astype(int)
Full Pipeline
python# Run complete analysis
python credit_default_pipeline.py

# This will:
# 1. Prepare non-transactional data
# 2. Perform feature selection
# 3. Run multi-seed variance analysis
# 4. Optimize hyperparameters with Optuna
# 5. Compare XGBoost vs Random Forest vs Logistic Regression
# 6. Generate visualizations and export results
Custom Analysis
pythonfrom pipeline import run_complete_stability_optimization

# Run with your own parameters
results = run_complete_stability_optimization(
    df=your_dataframe,
    target_col='F2_4_binary',
    features=top_30_features,
    n_variance_seeds=10,
    n_optuna_trials=50,
    optimization_seed=42,
    n_stability_seeds=10
)
Results
Model Performance
ModelPrecisionRecallF1-ScoreROC-AUCXGBoost (Tuned)0.5150.6500.5740.865Random Forest0.4870.6530.5580.842Logistic Regression0.4410.6710.5320.825
Business Impact

2.67x improvement in precision over baseline (19.3%)
Flags 1 in 2 at-risk individuals correctly (vs. 1 in 5 random)
Suitable for low-cost interventions (educational programs, proactive outreach)

Confusion Matrix (XGBoost)
                Predicted
              No    Yes
Actual No    1156   85
       Yes    131  169
Model Comparison
Strengths by Model
XGBoost:
 Highest precision and ROC-AUC
 Handles non-linear relationships well
 Built-in feature importance

Random Forest:
 More interpretable than XGBoost
 Robust to outliers
 Lower variance across seeds

Logistic Regression:
 Most interpretable (linear coefficients)
 Fastest training time
 Works well with standardized features

Limitations

Cross-Sectional Data: Cannot establish causality; longitudinal data needed
Self-Reported Data: Subject to recall and social desirability bias
Sample Scope: Limited to young adults (18-34) in the United States
Binary Outcome: Does not capture severity or frequency of defaults
Temporal Context: 2021 data includes pandemic-era economic conditions

Future Work
Short-Term

 Validate with 2024 NFCS data
 Test on older age groups (35+)
 Develop interpretability tools (SHAP values)
 Create interactive dashboard for risk screening

Long-Term

 Integrate real-time transactional data
 Longitudinal analysis (2-5 year tracking)
 Test intervention effectiveness (A/B testing)
 Expand to international populations
 Multi-class prediction (severity levels)


Contact
Author: Qihang Yang
Email: QYang1@my.harrisburgu.edu
LinkedIn: https://www.linkedin.com/in/andrewyangnb
Institution: Harrisburg University of Science and Technology

For questions, issues, or collaboration opportunities, please open an issue or submit a pull request.
