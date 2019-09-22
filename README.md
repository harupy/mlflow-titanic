# MLflow Titanic

A quick demo to show how MLflow works using the Titanic dataset.

![demo](https://user-images.githubusercontent.com/17039389/65383212-cc848280-dd4c-11e9-9f4a-16c8577e6622.gif)

## Getting Started

```
# create environment and activate it
conda env create -f environment.yml
conda activate mlflow-titanic

# preprocess data
python src/preprocess.py

# train model
python src/train.py
```

## Export Environment

```
conda env export | grep -v "^prefix: " > environment.yml
```