# Databricks notebook source
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# Load the table into a Spark DataFrame
df = spark.table("hive_metastore.default.test")


# COMMAND ----------

df_pd = df.toPandas()

# COMMAND ----------

# Rename all columns to follow _ format rather than spaces or dashes
df_pd.columns = df_pd.columns.str.replace(' ', '_').str.replace('-', '_')

display(df_pd)

# COMMAND ----------

df_pd.info()

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Engineering
# MAGIC

# COMMAND ----------

#Make Attrition variable numeric (Stayed is 0, Left is 1)

# COMMAND ----------

df_pd['Attrition'] = df_pd['Attrition'].map({'Stayed': 0, 'Left': 1})

# COMMAND ----------

#Bin Age column 

# COMMAND ----------

# Create quantile bins for the 'Age' column
df_pd['Age_Quantile'], bins = pd.qcut(df_pd['Age'], q=5, retbins=True)

# Format the bin ranges to remove decimals and create bin labels
df_pd['Age_Quantile'] = df_pd['Age_Quantile'].apply(
    lambda x: f"{int(x.left)}-{int(x.right)}"
)

# One-hot encode the quantile bins with bin ranges as column names
age_dummies = pd.get_dummies(df_pd['Age_Quantile'], prefix='Age')

# Concatenate the dummy variables back to the original DataFrame (optional)
df_pd = pd.concat([df_pd, age_dummies], axis=1)

df_pd = df_pd.drop(['Age', 'Age_Quantile'], axis=1)

# Display the resulting DataFrame
print(df_pd)

# COMMAND ----------

#Bin Monthly Income Column

# COMMAND ----------

# Create quantile bins for the 'Age' column
df_pd['Income_Quantile'], bins = pd.qcut(df_pd['Monthly_Income'], q=5, retbins=True)

# Format the bin ranges to remove decimals and create bin labels
df_pd['Income_Quantile'] = df_pd['Income_Quantile'].apply(
    lambda x: f"{int(x.left)}-{int(x.right)}"
)

# One-hot encode the quantile bins with bin ranges as column names
age_dummies = pd.get_dummies(df_pd['Income_Quantile'], prefix='Monthly_Income')

# Concatenate the dummy variables back to the original DataFrame (optional)
df_pd = pd.concat([df_pd, age_dummies], axis=1)

df_pd = df_pd.drop(['Monthly_Income', 'Income_Quantile'], axis=1)

# Display the resulting DataFrame
print(df_pd)

# COMMAND ----------

#Bin Years at Company Column

# COMMAND ----------

# Create quantile bins for the 'Age' column
df_pd['Years_at_Company_Quantile'], bins = pd.qcut(df_pd['Years_at_Company'], q=5, retbins=True)

# Format the bin ranges to remove decimals and create bin labels
df_pd['Years_at_Company_Quantile'] = df_pd['Years_at_Company_Quantile'].apply(
    lambda x: f"{int(x.left)}-{int(x.right)}"
)

# One-hot encode the quantile bins with bin ranges as column names
age_dummies = pd.get_dummies(df_pd['Years_at_Company_Quantile'], prefix='Years_at_Company')

# Concatenate the dummy variables back to the original DataFrame (optional)
df_pd = pd.concat([df_pd, age_dummies], axis=1)

df_pd = df_pd.drop(['Years_at_Company', 'Years_at_Company_Quantile'], axis=1)

# Display the resulting DataFrame
print(df_pd)

# COMMAND ----------

# Bin Distance from Home Column

# COMMAND ----------

# Create quantile bins for the 'Age' column
df_pd['Distance_from_Home_Quantile'], bins = pd.qcut(df_pd['Distance_from_Home'], q=5, retbins=True)

# Format the bin ranges to remove decimals and create bin labels
df_pd['Distance_from_Home_Quantile'] = df_pd['Distance_from_Home_Quantile'].apply(
    lambda x: f"{int(x.left)}-{int(x.right)}"
)

# One-hot encode the quantile bins with bin ranges as column names
age_dummies = pd.get_dummies(df_pd['Distance_from_Home_Quantile'], prefix='Distance_from_Home')

# Concatenate the dummy variables back to the original DataFrame (optional)
df_pd = pd.concat([df_pd, age_dummies], axis=1)

df_pd = df_pd.drop(['Distance_from_Home', 'Distance_from_Home_Quantile'], axis=1)

# Display the resulting DataFrame
print(df_pd)

# COMMAND ----------

# One Hot Encode the Job Role Column

# COMMAND ----------

# One Hot Encode the Job Role Column
job_role_dummies = pd.get_dummies(df_pd['Job_Role'], prefix='Job_Role')

# Concatenate the one-hot encoded columns back to the original DataFrame
df_pd = pd.concat([df_pd, job_role_dummies], axis=1).drop(columns=['Job_Role'])

print(df_pd)

# COMMAND ----------

#Label encode multiple features

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

# Columns to label encode
columns_to_encode = ['Work_Life_Balance', 'Job_Satisfaction', 'Performance_Rating', 'Education_Level', 'Job_Level', 'Company_Size', 'Company_Reputation', 'Employee_Recognition']
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to each column and save as new columns with the suffix "_Encoded"
for column in columns_to_encode:
    df_pd[column + '_Encoded'] = label_encoder.fit_transform(df_pd[column])

display(df_pd)
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to each column
for column in columns_to_encode:
    df_pd[column] = label_encoder.fit_transform(df_pd[column])

print(df_pd)

# COMMAND ----------

# One hot encode specified columns
columns_to_one_hot_encode = ['Gender', 'Overtime', 'Marital_Status', 'Remote_Work', 'Leadership_Opportunities', 'Innovation_Opportunities']

# Apply one hot encoding using pandas get_dummies
df_pd = pd.get_dummies(df_pd, columns=columns_to_one_hot_encode)

print(df_pd)

# COMMAND ----------

display(df_pd)

# COMMAND ----------

df_final = df_pd.drop(
    columns=[
        'Company_Tenure',
        'Work_Life_Balance', 
        'Job_Satisfaction', 
        'Performance_Rating', 
        'Education_Level', 
        'Job_Level', 
        'Company_Size', 
        'Company_Reputation', 
        'Employee_Recognition'
    ]
)

# COMMAND ----------

print(df_final)

# COMMAND ----------

display(df_final)

# COMMAND ----------

# Create Feature Store

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

df_spark = spark.createDataFrame(df_final)

# Initialize Feature Store client
fs = FeatureStoreClient()

# Define the new feature table name
feature_table_name = 'employee_attrition_processed_features_test'

# Create the feature table in the feature store
fs.create_table(
    name=feature_table_name,
    primary_keys=["Employee_ID"],  # Replace "Employee_ID" with your actual primary key column name
    df=df_spark,
    description="Feature store table created from employee attrition data with processed features"
)

# Write the Spark DataFrame to the feature store
fs.write_table(
    name=feature_table_name,
    df=df_spark,
    mode="overwrite"
)