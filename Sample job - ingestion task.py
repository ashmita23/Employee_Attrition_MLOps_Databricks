# Databricks notebook source
# Let's see what example data sets exist
display(dbutils.fs.ls('/databricks-datasets'))

# COMMAND ----------

# Let's setup the lending example data for ingestion
d_path="/databricks-datasets/samples/lending_club/parquet/"
l_data = spark.read.parquet(d_path)

# Overview of the schema
display(l_data.printSchema())

# Save example data in the loans table and overwrite it if it already exists
l_data.write.mode("overwrite").saveAsTable("loans")


# COMMAND ----------

# Let's take a look at the structure of the loans data
loans = spark.read.format("delta").table("loans")
loans_desc = loans.describe().toPandas()
loans_count = loans.count()
dbutils.jobs.taskValues.set(key = "count", value = int(loans_count))
display(loans_count)
display(loans_desc)