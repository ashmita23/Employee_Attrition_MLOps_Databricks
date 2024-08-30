# Databricks notebook source
# MAGIC %pip install hyperopt optuna xgboost
# MAGIC

# COMMAND ----------

import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()
feature_table = "default.employee_attrition_processed_features_4"

# Load the feature table as a DataFrame
features_df = fs.read_table(feature_table)

# COMMAND ----------

features_df

# COMMAND ----------

type(features_df)
features_df = features_df.toPandas()

# COMMAND ----------

X = features_df.drop(columns=["Attrition","Employee_ID"])
y = features_df["Attrition"]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def objective(params):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'colsample_bytree': params['colsample_bytree'],
        'learning_rate': params['learning_rate'],
        'max_depth': int(params['max_depth']),
        'min_child_weight': params['min_child_weight'],
        'n_estimators': int(params['n_estimators']),
        'random_state': 109765561
    }
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=109765561)
    
    # Train the model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predict on the validation set
    preds = model.predict(X_val)
    pred_probs = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, pred_probs)
    
    # Log metrics for comparison
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, ROC AUC: {roc_auc}")
    
    # We aim to minimize the negative F1-score (or you can use other metrics)
    return {'loss': -f1, 'status': STATUS_OK, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}



# COMMAND ----------

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=109765561)


# COMMAND ----------


trials = Trials()

best_params = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,  # Number of iterations to run
    trials=trials
)

# Extract best trial metrics
best_trial = trials.best_trial['result']
print(f"Best trial metrics: {best_trial}")



# COMMAND ----------

best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])
# Extract the best trial metrics from hyperopt
best_trial = trials.best_trial['result']
accuracy = best_trial['accuracy']
precision = best_trial['precision']
recall = best_trial['recall']
f1 = best_trial['f1']
roc_auc = best_trial['roc_auc']

# Log the metrics with MLflow
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.xgboost.log_model(final_model, "model")
    mlflow.end_run()

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.xgboost.log_model(final_model, "model")
    mlflow.end_run()

# COMMAND ----------



# Predict on the validation/test set
y_pred = final_model.predict(X_val)

# Create confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=final_model.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# COMMAND ----------

from sklearn.metrics import roc_curve, auc

# Predict probabilities
y_pred_proba = final_model.predict_proba(X_val)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# COMMAND ----------

import xgboost as xgb

# Plot feature importance
xgb.plot_importance(final_model,importance_type='weight', max_num_features=10)
plt.title('Feature Importance')
plt.show()

# COMMAND ----------

fig, ax = plt.subplots()

# Plot top 10 feature importances
xgb.plot_importance(final_model, importance_type='weight', max_num_features=10, ax=ax)

# Customize the plot to make grid lines transparent
ax.grid(False)  # Disable the grid lines
ax.set_facecolor('none')  # Set the background to be transparent

plt.title('Top 10 Feature Importance')
plt.show()

# COMMAND ----------

from sklearn.metrics import classification_report

# Ensure target_names is an array of strings
target_names_str = [str(cls) for cls in final_model.classes_]

# Generate classification report
report = classification_report(y_val, y_pred, target_names=target_names_str)
print(report)

# COMMAND ----------

# Get all parameters
params = final_model.get_params()

# Filter out parameters with None values and round the others to 2 decimal places
non_none_params = {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items() if v is not None}

print("Best Parameters chosen by Hyperopt for XGBoost:")
for key, value in non_none_params.items():
    print(f"{key}: {value}")


# COMMAND ----------

