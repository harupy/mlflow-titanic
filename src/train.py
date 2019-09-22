
from plot_funcs import (plot_confusion_matrix,
                        plot_corr_matrix,
                        plot_feature_importance,
                        plot_metric,
                        plot_roc_curve,
                        plot_pr_curve,
                        plot_scores,
                        log_plot,
                        )
from config import PROCESSED_TRAIN_PATH
import mlflow
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             precision_recall_curve)
import numpy as np
import pandas as pd
import os
import warnings
import pickle
from collections import defaultdict
warnings.filterwarnings('ignore')


def devide_by_sum(x):
  return x / x.sum()


def get_scores(y_true, y_pred):
  return {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
  }


def print_devider(title):
  print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))


def train_model(X, y, params, exp_path):
  fold_params = params['fold']
  model_params = params['model']
  fit_params = params['fit']

  # set mlflow experiment
  try:
    mlflow.create_experiment(exp_path)
  except (mlflow.exceptions.RestException, mlflow.exceptions.MlflowException):
    print('The specified experiment ({}) already exists.'.format(exp_path))

  mlflow.set_experiment(exp_path)

  skf = StratifiedKFold(**fold_params)
  models = []
  metrics = []

  y_proba = np.zeros(len(X))
  y_pred = np.zeros(len(X))

  feature_importances_split = np.zeros(X.shape[1])
  feature_importances_gain = np.zeros(X.shape[1])

  scores = defaultdict(int)

  with mlflow.start_run() as run:
    corr = pd.concat((X, y), axis=1).corr()
    log_plot(corr, plot_corr_matrix, 'correlation_matrix.png')

    for fold_no, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
      print_devider(f'Fold: {fold_no}')

      X_train, X_valid = X.iloc[idx_train, :], X.iloc[idx_valid, :]
      y_train, y_valid = y.iloc[idx_train], y.iloc[idx_valid]

      # train model
      model = lgbm.LGBMClassifier(**model_params)
      model.fit(X_train, y_train, **fit_params, eval_set=[(X_valid, y_valid)], eval_names=['valid'])
      metrics.append({
        'name': model.metric,
        'values': model.evals_result_['valid'][model.metric],
        'best_iteration': model.best_iteration_
      })
      models.append(model)

      # feature importance
      feature_importances_split += devide_by_sum(model.booster_.feature_importance(importance_type='split')) / skf.n_splits
      feature_importances_gain += devide_by_sum(model.booster_.feature_importance(importance_type='gain')) / skf.n_splits

      # predict
      y_valid_proba = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
      y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
      y_proba[idx_valid] = y_valid_proba
      y_pred[idx_valid] = y_valid_pred

      # evaluate
      scores_valid = get_scores(y_valid, y_valid_pred)

      mlflow.log_metrics({
        **scores_valid,
        'best_iteration': model.best_iteration_,
      }, step=fold_no)

      print('\nScores')
      print(scores_valid)

      # record scores
      for k, v in scores_valid.items():
        scores[k] += v / skf.n_splits

    # log training parameters
    mlflow.log_params({
      **fold_params,
      **model_params,
      **fit_params,
      'cv': skf.__class__.__name__,
      'model': model.__class__.__name__
    })

    print_devider('Saving plots')

    # scores
    log_plot(scores, plot_scores, fp='scores.png')

    # feature importance
    features = np.array(model.booster_.feature_name())
    log_plot((features, feature_importances_split, 'Feature Importance: split'),
             plot_feature_importance, 'feature_importance_split.png')
    log_plot((features, feature_importances_gain, 'Feature Importance: gain'),
             plot_feature_importance, 'feature_importance_gain.png')

    # train history
    log_plot(metrics, plot_metric, 'metric_history.png')

    # confusion matrix
    cm = confusion_matrix(y, y_pred)
    log_plot(cm, plot_confusion_matrix, 'confusion_matrix.png')

    # roc curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    log_plot((fpr, tpr), plot_roc_curve, 'roc_curve.png')

    # precision-recall curve
    pre, rec, _ = precision_recall_curve(y, y_proba)
    log_plot((pre, rec), plot_pr_curve, 'pr_curve.png')

    # pickle trained models
    models_path = 'models.pkl'
    with open(models_path, 'wb') as f:
      pickle.dump(models, f)
      mlflow.log_artifact(models_path)
      mlflow.log_param('model_path', os.path.join(run.info.artifact_uri, models_path))
      os.remove(models_path)


def main():
  print(os.listdir('data'))
  train = pd.read_pickle(PROCESSED_TRAIN_PATH)

  X = train.drop('Survived', axis=1)
  y = train['Survived']

  SEED = 0

  params = {
    'model': {
      'objective': 'binary',
      'metric': 'auc',
      'n_estimators': 100000,
      'learning_rate': 0.05,
      'random_state': SEED,
      'n_jobs': -1
    },

    'fit': {
      'early_stopping_rounds': 100,
      'verbose': 10
    },

    'fold': {
      'n_splits': 5,
      'shuffle': True,
      'random_state': SEED
    }
  }

  train_model(X, y, params, 'titanic')
  print_devider('MLFlow UI (Ctrl-C to quit)')
  os.system('mlflow ui')


if __name__ == '__main__':
  main()
