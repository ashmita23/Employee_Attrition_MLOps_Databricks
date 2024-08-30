# Databricks notebook source
from databricks.feature_store import FeatureStoreClient


# COMMAND ----------

fs = FeatureStoreClient()


# COMMAND ----------

feature_table = "default.employee_attrition_processed_features_4"

# Load the feature table as a DataFrame
features_df = fs.read_table(feature_table)

# COMMAND ----------

import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# COMMAND ----------

features_df

# COMMAND ----------

type(features_df)
features_df = features_df.toPandas()

# COMMAND ----------

X = features_df.drop(columns=["Attrition","Employee_ID"])
y = features_df["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

param_dist = {
    'colsample_bytree': np.linspace(0.4, 0.5, 10),
    'lambda_l1': np.linspace(5, 15, 10),
    'lambda_l2': np.linspace(70, 100, 10),
    'learning_rate': np.linspace(0.5, 0.8, 10),
    'max_bin': np.arange(300, 400, 10),
    'max_depth': np.arange(4, 7),
    'min_child_samples': np.arange(60, 100, 5),
    'n_estimators': np.arange(50, 80, 5),
    'num_leaves': np.arange(10, 20),
    'path_smooth': np.linspace(10, 15, 10),
    'subsample': np.linspace(0.5, 0.6, 10),
    'random_state': [109765561]
}

# COMMAND ----------

model = lgb.LGBMClassifier()

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  
    scoring='f1_weighted', 
    cv=3,  
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# COMMAND ----------

best_model = random_search.best_estimator_

# Log the model with MLflow
with mlflow.start_run():
    # Log the best hyperparameters
    mlflow.log_params(random_search.best_params_)

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Calculate and log metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log the best model
    mlflow.lightgbm.log_model(best_model, "model")

    # End the MLflow run
    mlflow.end_run()

# COMMAND ----------

