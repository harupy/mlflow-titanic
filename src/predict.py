# Databricks notebook source
# MAGIC %md
# MAGIC # TOC
# MAGIC ---
# MAGIC - [Install MLflow](#notebook/2069219/command/2069823)
# MAGIC - [Query Best Model](#notebook/2069219/command/2069826)
# MAGIC - [Display Plots](#notebook/2069219/command/2069828)
# MAGIC - [Load Trained Models](#notebook/2069219/command/2069830)
# MAGIC - [Wrap Models as One Model](#notebook/2069219/command/2069832)
# MAGIC - [Predict with Test Data](#notebook/2069219/command/2069834)

# COMMAND ----------

# MAGIC %md
# MAGIC # Install MLflow

# COMMAND ----------

import pickle
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
from functools import reduce
import re
import pandas as pd
import numpy as np
import mlflow
dbutils.library.installPyPI('mlflow')

# COMMAND ----------


mlflow.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC # Query Best Model

# COMMAND ----------

pd.set_option('display.max_columns', 1000)

EXPERIMENT_ID = 2054794

# query the most accurate & latest model
runs = mlflow.search_runs(EXPERIMENT_ID, order_by=['metrics.accuracy DESC', 'attribute.start_time DESC'], max_results=10)
runs

# COMMAND ----------

# MAGIC %md
# MAGIC # Display Plots

# COMMAND ----------


# TODO: sort figure order
def convert_dbfs(path):
  return re.sub(r'^dbfs:', '/dbfs', path)


def read_image_as_base64(fp):
  im = Image.open(fp)
  buffer = io.BytesIO()
  im.save(buffer, format='PNG')
  buffer.seek(0)
  return base64.b64encode(buffer.read()).decode('ascii')


plots = [convert_dbfs(f.path) for f in dbutils.fs.ls(runs.loc[0, 'artifact_uri']) if f.path.endswith('.png')]
img_tag = '\n<img src="data:image/png;base64,{}" width="600px" style="border: 1px solid silver; padding: 0px; margin: 0px;">'
html = reduce(lambda acc, p: acc + img_tag.format(read_image_as_base64(p)), plots, '')
displayHTML('<h1>Dashboard</h1>' + html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Trained Models

# COMMAND ----------


def load_model(model_path):
  with open(model_path, 'rb') as f:
    return pickle.load(f)


model_path = convert_dbfs(runs.loc[0, 'params.model_path'])
models = load_model(model_path)
models

# COMMAND ----------

# MAGIC %md
# MAGIC # Wrap Models as One Model

# COMMAND ----------


class EnsembleModel:
  def __init__(self, models):
    self.models = models

  def predict_proba(self, X):
    proba = np.zeros((len(X), models[0].n_classes_))
    for model in models:
      proba += model.predict_proba(X, num_iteration=model.best_iteration_)
    return proba / len(models)


model = EnsembleModel(models)

# COMMAND ----------

# MAGIC %md
# MAGIC # Predict with Test Data

# COMMAND ----------

X_test = pd.read_csv('/dbfs/FileStore/tables/harutaka.kawamura/titanic/test_clean.csv')
proba = model.predict_proba(X_test)
proba

# COMMAND ----------
