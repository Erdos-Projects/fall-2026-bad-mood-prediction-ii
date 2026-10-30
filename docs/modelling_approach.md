
# Modeling Approach

- **Objective:** Predict the probability of seven next-day moods—ALERT, HAPPY, NEUTRAL, RESTED/RELAXED, SAD, TENSE/ANXIOUS, and TIRED—using seven independent binary classifiers (mood labels can overlap).

- **Data preparation:** Construct next-day targets by matching each participant's mood report on day *t + 1* to their features on day *t*. This yields **2,276 observations** from **63 participants**.

- **Features (20 total):** Ten wearable/physiological measures (activity, heart rate, and sleep), three demographic features (age group, gender, BMI group), and seven current-day mood indicators. Impute missing numerical values; encode categorical variables and missing mood reports separately. Fit preprocessing only on training data.

- **Validation design:** Split observations chronologically **within each participant** into **1,815 development** and **461 held-out test** records. Use three expanding-window, participant-wise chronological folds on the development set for model selection.

- **Algorithm comparison:** Benchmark **Logistic Regression, KNN, XGBoost, and Random Forest** on identical features and folds. 

- **Hyperparameter tuning:** The selected algorithm from previous step is further tuned and used downstream.

- **Classification thresholds:** With the tuned algorithm, choose a separate probability cutoff for each mood by maximizing F1 on development cross-validation predictions. Freeze thresholds before final testing.

- **Final evaluation:** Refit seven tuned algorithm on all train data and evaluate once on the held-out test set using **ROC-AUC, AP, Brier score, precision, recall, and F1**. 

- **Interpretation:** Use test-set permutation importance (measured by decrease in AP). 


