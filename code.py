import pandas as pd
import os

# 📁 File paths for the dataset
train_path = r"C:\Users\GorkemKoskaya\Desktop\banking\train.csv"
test_path = r"C:\Users\GorkemKoskaya\Desktop\banking\test.csv"

# 📥 Load the train and test datasets
df_train = pd.read_csv(train_path, sep=";")
df_test = pd.read_csv(test_path, sep=";")

# 👀 Preview the first few rows
print("🔎 Train set preview:")
print(df_train.head())

print("\n🔎 Test set preview:")
print(df_test.head())

# 📏 Dataset shapes (rows, columns)
print("\n📐 Dataset Dimensions:")
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

# 🧾 Data types and missing values
print("\nℹ️ Train Dataset Info:")
print(df_train.info())

# 🎯 Target variable distribution
print("\n📊 Target Variable Distribution (y):")
print(df_train["y"].value_counts())
print("\n📊 Target Class Ratios (%):")
print(df_train["y"].value_counts(normalize=True) * 100)

# 🔄 Encode the target variable ('yes' -> 1, 'no' -> 0)
df_train["y"] = df_train["y"].map({"yes": 1, "no": 0})

# ⛔ Drop the 'duration' column as it may cause data leakage
df_train = df_train.drop("duration", axis=1)

# 🔁 One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_train, drop_first=True)

# 🎯 Split into features and target
X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]

# ✂️ Train-test split (80% train, 20% test, stratified)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ⚙️ Import model and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 📌 Logistic Regression Model
model_lg = LogisticRegression(max_iter=1000, random_state=42)
model_lg.fit(X_train, y_train)

# 🔮 Predictions
y_pred_lg = model_lg.predict(X_test)

# ✅ Performance Evaluation
print("📈 Logistic Regression Train Accuracy:", model_lg.score(X_train, y_train))
print("📉 Logistic Regression Test Accuracy:", model_lg.score(X_test, y_test))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lg))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_lg))


# 🌲 Random Forest Model
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# ✅ Performance Evaluation
print("📈 Random Forest Train Accuracy:", model_rf.score(X_train, y_train))
print("📉 Random Forest Test Accuracy:", model_rf.score(X_test, y_test))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_rf))


# 🚀 XGBoost Model
import xgboost as xgb

model_xgb = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# ✅ Performance Evaluation
print("📈 XGBoost Train Accuracy:", model_xgb.score(X_train, y_train))
print("📉 XGBoost Test Accuracy:", model_xgb.score(X_test, y_test))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_xgb))


# 🔍 Support Vector Machine (SVM) Model
from sklearn.svm import SVC

model_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)

# ✅ Performance Evaluation
print("📈 SVM Train Accuracy:", model_svm.score(X_train, y_train))
print("📉 SVM Test Accuracy:", model_svm.score(X_test, y_test))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_svm))


# 🧠 Naive Bayes Model
from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

# ✅ Performance Evaluation
print("📈 Naive Bayes Train Accuracy:", model_nb.score(X_train, y_train))
print("📉 Naive Bayes Test Accuracy:", model_nb.score(X_test, y_test))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_nb))

# 📊 Compare Model Performance Metrics

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define model names and their corresponding predictions
model_names = ['LogReg', 'RandomForest', 'XGBoost', 'SVM', 'NaiveBayes']
y_preds = [y_pred_lg, y_pred_rf, y_pred_xgb, y_pred_svm, y_pred_nb]

# Dictionary to store evaluation metrics
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": []
}

# Compute metrics for each model
for name, y_pred in zip(model_names, y_preds):
    results["Model"].append(name)
    results["Accuracy"].append(accuracy_score(y_test, y_pred))
    results["Precision"].append(precision_score(y_test, y_pred, pos_label=1))
    results["Recall"].append(recall_score(y_test, y_pred, pos_label=1))
    results["F1-Score"].append(f1_score(y_test, y_pred, pos_label=1))

# Convert results to DataFrame and set index
df_metrics = pd.DataFrame(results)
df_metrics.set_index("Model", inplace=True)

# 📈 Visualize model comparison
df_metrics.plot(kind='bar', figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 📌 Summary statistics for SVM
import numpy as np
print("Unique predicted classes and their counts:", np.unique(y_pred_svm, return_counts=True))
print(classification_report(y_test, y_pred_svm))

# ⚖️ Apply SMOTE for Class Imbalance

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 🔧 Hyperparameter Tuning for XGBoost

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

grid_search_xgb = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='f1',
    verbose=2,
    n_jobs=-1
)

grid_search_xgb.fit(X_train_resampled, y_train_resampled)
best_xgb = grid_search_xgb.best_estimator_

# 🔧 Hyperparameter Tuning for Naive Bayes

from sklearn.naive_bayes import GaussianNB

param_grid_nb = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]}
grid_search_nb = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=param_grid_nb,
    cv=3,
    scoring='f1',
    verbose=2
)

grid_search_nb.fit(X_train_resampled, y_train_resampled)
best_nb = grid_search_nb.best_estimator_

# 🧪 Model Evaluation

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

# Predictions and Probabilities
y_pred_xgb = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

y_pred_nb = best_nb.predict(X_test)
y_proba_nb = best_nb.predict_proba(X_test)[:, 1]

# 📌 XGBoost Evaluation
print("📌 XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba_xgb))

# 📌 Naive Bayes Evaluation
print("\n📌 Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba_nb))

# 📈 ROC Curve Visualization
fig, ax = plt.subplots()
RocCurveDisplay.from_estimator(best_xgb, X_test, y_test, ax=ax, name="XGBoost")
RocCurveDisplay.from_estimator(best_nb, X_test, y_test, ax=ax, name="Naive Bayes")
plt.title("ROC Curve Comparison")
plt.show()

# 📉 Precision-Recall Curve Visualization
fig, ax = plt.subplots()
PrecisionRecallDisplay.from_estimator(best_xgb, X_test, y_test, ax=ax, name="XGBoost")
PrecisionRecallDisplay.from_estimator(best_nb, X_test, y_test, ax=ax, name="Naive Bayes")
plt.title("Precision-Recall Curve Comparison")
plt.show()

# 🔍 Hyperparameter Tuning for XGBoost with Class Imbalance Consideration

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Ensure X_train and y_train are defined before this block

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'scale_pos_weight': [1, 5, 10]  # Important for handling class imbalance
}

xgb = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 🏆 Best model and performance
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 📌 Prediction on Test Set
y_pred = best_model.predict(X_test)

# 📊 Feature Importance

importances = best_model.feature_importances_

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Create DataFrame for importances
feature_names = X_train.columns
feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.show()

# 🧠 SHAP Values (Optional - continuation expected)
import shap
import matplotlib.pyplot as plt

# 🔍 SHAP Analysis for Feature Importance
# TreeExplainer is used to compute SHAP values
explainer = shap.TreeExplainer(best_model)  # Using the best model from GridSearch
shap_values = explainer.shap_values(X_test)

# Summary plot (bar): Displays overall feature impact
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Full summary plot: SHAP value distribution
shap.summary_plot(shap_values, X_test)

# 📌 Feature Subset Selection based on SHAP
selected_features = [
    'contact_unknown',
    'poutcome_success',
    'housing_yes',
    'day',
    'balance',
    'age',
    'marital_married',
    'campaign',
    'month_aug',
    'month_may',
    'month_jul',
    'loan_yes'
]

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Train-test split on selected features
X_selected = X[selected_features]
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train XGBoost model on selected features
model_xgb_sel = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model_xgb_sel.fit(X_train_sel, y_train_sel)

# Predictions
y_pred_sel = model_xgb_sel.predict(X_test_sel)
y_proba_sel = model_xgb_sel.predict_proba(X_test_sel)[:, 1]

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test_sel, y_pred_sel))

print("\nClassification Report:")
print(classification_report(y_test_sel, y_pred_sel))

print("ROC AUC:", round(roc_auc_score(y_test_sel, y_proba_sel), 3))


# ⚖️ Handling Class Imbalance using SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sel, y_train_sel)

from collections import Counter
print("Resampled Class Distribution:", Counter(y_train_sm))

# Train new XGBoost model on SMOTE-balanced data
model_xgb_sm = XGBClassifier(random_state=42)
model_xgb_sm.fit(X_train_sm, y_train_sm)

# Predictions
y_pred_sm = model_xgb_sm.predict(X_test_sel)
y_proba_sm = model_xgb_sm.predict_proba(X_test_sel)[:, 1]

# 🔧 Optimal Threshold Selection using ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test_sel, y_proba_sm)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

print("Optimal threshold:", round(optimal_threshold, 3))

# Apply custom threshold
y_pred_thresh = (y_proba_sm >= optimal_threshold).astype(int)

# Final Evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

print("Confusion Matrix:\n", confusion_matrix(y_test_sel, y_pred_thresh))
print("\nClassification Report:\n", classification_report(y_test_sel, y_pred_thresh))
print("ROC AUC:", round(roc_auc_score(y_test_sel, y_proba_sm), 3))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Predicted probabilities for the positive class
y_scores = y_proba_xgb_sm

# Compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_test_sel, y_scores)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC Curve', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve & Threshold Analysis')
plt.grid(True)

# Highlight the optimal point based on Youden's Index
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_thresh = thresholds[best_idx]

plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best Threshold = {best_thresh:.2f}')
plt.legend()
plt.show()

print(f"🏷️ Best threshold (based on Youden's Index): {best_thresh:.2f}")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Define a custom threshold
new_thresh = 0.31

# Apply threshold to obtain final predictions
y_pred_thresh_new = (y_proba_xgb_sm >= new_thresh).astype(int)

# Final evaluation with custom threshold
print("Confusion Matrix:\n", confusion_matrix(y_test_sel, y_pred_thresh_new))
print("\nClassification Report:\n", classification_report(y_test_sel, y_pred_thresh_new))
print("ROC AUC:", round(roc_auc_score(y_test_sel, y_proba_xgb_sm), 3))  # AUC is calculated using probabilities


