import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import argparse
import pandas as pd
import pickle
import mlflow.sklearn
import lightgbm as lgb
import yaml


def fit_lgbm_sklearn_crossvalidator(split_prop):
    """
    Helper function that fits a scikit-learn 5-fold cross validated model
    to predict a binary label `target` on the passed-in training DataFrame
    using the columns in `features`
    :param: split_prop: float, percent of data to use as training in train_test_split method
    """

    local_dir = os.path.abspath(os.path.dirname(__file__))
    training_config_path = os.path.join(local_dir, '../../config/training.yaml')
    with open(training_config_path) as conf:
        training_config = yaml.full_load(conf)

    processed_data_dir = os.path.join(local_dir, '../../data/processed')
    features = training_config['features']
    target = training_config['target']
    seed = 7

    train_csv = os.path.join(processed_data_dir, 'train.csv')
    train_df = pd.read_csv(train_csv)

    X = train_df[features]
    y = train_df[target]
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=split_prop, random_state=seed)

    mlflow.log_metric("training_nrows", len(X_train))
    mlflow.log_metric("test_nrows", len(X_test))
    print('Training: {0}, test: {1}'.format(len(X_train), len(X_test)))

    lgbm = lgb.LGBMClassifier()
    param_space = training_config['parameter_space']
    param_grid = {'__max_depth': param_space['max_depth'],
                  '__min_split_gain': param_space['gamma'],
                  '__min_child_weight': param_space['min_child_weight'],
                  '__learning_rate': param_space['learning_rate'],
                  '__colsample_bytree': param_space['colsample_bytree'],
                  '__num_leaves': param_space['num_leaves']
                  }

    # define pipeline and initialize RandomizedSearchCV
    steps = [('imputation', SimpleImputer()),
             ('', lgbm)]
    pipeline = Pipeline(steps)
    crossval = RandomizedSearchCV(pipeline,
                                  param_grid,
                                  scoring='roc_auc',
                                  n_iter=50,
                                  random_state=seed,
                                  cv=5)
    mlflow.log_param("split_prop", split_prop)
    # fit and log best estimator
    cvModel = crossval.fit(X_train,
                           y_train.values.ravel())
    pickle.dump(cvModel,
                open(os.path.join(local_dir, '../../models/lgbm.pkl'), 'wb'))
    mlflow.sklearn.log_model(cvModel.best_estimator_,
                             "best-5-fold-cross-validated-{}"
                             .format(str(lgbm).split('(')[0]),
                             conda_env=os.path.join(local_dir, '../../config/conda.yaml'))
    y_pred = cvModel.best_estimator_.predict(X_test)
    print('Train ROC: {:.3f}'.format(cvModel.best_score_))
    test_roc = roc_auc_score(y_test, y_pred)
    print('Test ROC: {:.3f}'.format(test_roc))
    # log metric(s), if multiple can be as a dict
    mlflow.log_metrics({"ROC_AUC_train": cvModel.best_score_,
                       "ROC_AUC_test": test_roc})
    # log parameters of best fit model
    for key in cvModel.best_params_:
        mlflow.log_param(key, cvModel.best_params_[key])

    return cvModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split_prop", default=0.8, type=float)
    args = parser.parse_args()

    fit_lgbm_sklearn_crossvalidator(args.split_prop)
