ML titanic challenge
==============================

A MLflow training pipeline and API for binary classification of survival for passengers aboard the Titanic

Project Organization
------------

    ├── README.md                <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models                   <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                <- Jupyter notebooks.
    │
    ├── references               <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                 <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                      <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- 
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   │                 
    │   │   ├── predict_lgbm.py
    │   │   └── train_lgbm.py



--------

MLflow Multistep Workflow
--------------------------
This multistep workflow was modified from the original example to run inside of a 
Databricks notebook. Most of this readme remains from the original example, but includes 
added support for Databricks.


There are two simple steps to this multistep workflow:

- **build_features.py**: Loads the raw Titanic training data from /data/raw/train.csv, 
  one hot encodes categorical columns, drops unimportant columns for training, and saves
  processed data to /data/processed/train.csv.

- **train_lgbm.py**: Trains a LightGBM model and logs various things such as
  the best parameters chosen by RandomizedSearchCV, the best model fitted in
  pickle format, roc_auc metric, etc.

While we can run each of these steps manually, here we have a driver
run, defined as **main** (main.py) to automate the entire pipeline. This run will run
all the steps (defined in the MLproject file) in order, passing the results of one to the 
next, and profile the CPU and RAM usage of the entire pipeline. Additionally, this run will
attempt to determine if a sub-run has already been executed successfully with the same 
parameters and, if so, reuse the cached results.

Running this Example
^^^^^^^^^^^^^^^^^^^^

**In your local machine**:

First install requirements with conda:
`$ conda create --name <env> --file requirements.txt

(Note: Requirements installed for mac m1 ARM architecture and may not be compatible on other architectures)
`
In order for the multistep workflow to find the other steps, you must execute 
``mlflow run .`` from the root of the project. You can change the parameter space, features
to use, and target variable to predict in /config/training.yaml and the compare the results
of various iterations by runnning ``mlflow ui`` in the root of the directory.
    
**In Databricks community edition** (using Databricks runtime version 7.0 ML):

First you need to setup your credentials

.. code-block:: python

    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://community.cloud.databricks.com\ntoken = "+token,overwrite=True)
    
then once uploaded as a repo to github you can go ahead and execute the project with the MLflow api:

.. code-block:: python

    mlflow.run('git://github.com/<user>/rappi_challenge.git')

Running Flask API
----------------------------------

Running the API
^^^^^^^^^^^^^^^^^^^^

**In your local machine**:

First install requirements with conda:
`$ conda create --name <env> --file requirements.txt

(Note: Requirements installed for mac m1 ARM architecture and may not be compatible on other architectures)

From root directory run the application server on port 2000:
`python -m flask run --port=2000`

From another terminal in root directory send requests to the server 
(using the training data for example purposes only) with:
`python request.py`

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# ml-titanic-challenge
