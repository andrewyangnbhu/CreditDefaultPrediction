# CreditDefaultPrediction
Repository for Predicting Credit Card Default in Young Adults: A Machine Learning Approach Using Non-Transactional Survey Data

Credit Card Default Prediction
Predicting credit card default risk among young adults using non-transactional behavioral data from the 2021 National Financial Capability Study.
Key Results

86.5% ROC-AUC using XGBoost
51.5% Precision at 65% recall (2.67x improvement over baseline)
Payday loan usage identified as strongest predictor

Dataset

Source: 2021 NFCS
Sample: 7,705 young adults (ages 18-34)
Features: 30 non-transactional predictors
Target: Credit card late fees (19.3% prevalence)

Installation
bashpip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn
Usage
python# Run complete pipeline
python credit_default_pipeline.py

# Load trained model
import pickle
with open('best_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
Model Performance
ModelPrecisionRecallF1ROC-AUCXGBoost0.5150.6500.5740.865Random Forest0.4870.6530.5580.842Logistic Reg0.4410.6710.5320.825
Methodology

Data preprocessing (handle missing values, encode categoricals)
XGBoost-based feature selection (top 30 features)
Hyperparameter optimization using Optuna (100 trials)
Decision threshold tuning for precision-recall balance
Model comparison (XGBoost vs RF vs LR)

Limitations

Cross-sectional data (no causality)
Self-reported survey responses
Limited to young adults (18-34)
Binary outcome (doesn't capture severity)


Contact
Author: Qihang Yang
Email: QYang1@my.harrisburgu.edu
LinkedIn: https://www.linkedin.com/in/andrewyangnb
Institution: Harrisburg University of Science and Technology

For questions, issues, or collaboration opportunities, please open an issue or submit a pull request.
