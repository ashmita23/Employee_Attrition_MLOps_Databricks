# Databricks notebook source
!pip install evidently

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %pip install --upgrade evidently

# COMMAND ----------

!pip install evidently==0.2.4  # Replace with a version that works for your environment


# COMMAND ----------

from evidently.report import Report