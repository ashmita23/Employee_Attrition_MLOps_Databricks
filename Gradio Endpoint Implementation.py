# Databricks notebook source
pip install gradio

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import gradio as gr
import mlflow
import mlflow.xgboost
import xgboost as xgb
from databricks.feature_store import FeatureStoreClient
import pandas as pd


# COMMAND ----------

model_name = "Final_Attrition_XGBoost"
model_version = "1"

# Load the model
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# COMMAND ----------

fs = FeatureStoreClient()


# COMMAND ----------

feature_table = "default.employee_attrition_processed_features_4"
features_df = fs.read_table(feature_table)

# Convert to the format expected by your model
input_data = features_df.drop("attrition").toPandas()

# COMMAND ----------

assert isinstance(input_data, pd.DataFrame), "Input data is not a Pandas DataFrame"

# COMMAND ----------

def predict():
    try:
        # Ensure input_data is a DataFrame
        if not isinstance(input_data, pd.DataFrame):
            return "Error: Input data is not a DataFrame."

        # Perform prediction directly on the Pandas DataFrame
        prediction = model.predict(input_data.iloc[[0]])

        mapped_prediction = ["Stayed" if pred == 0 else "Left" for pred in prediction]
        
        return mapped_prediction  # Convert prediction to a list for easier display
    except Exception as e:
        return f"Error: {str(e)}"
# Set up Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=None,  # No manual inputs, data comes from the feature store
    outputs=gr.Textbox(label="Prediction"),
    title="Employee Attrition Predictor",
    description="Click the button to predict employee attrition using processed features."
)

# Launch the app
interface.launch(share=True)