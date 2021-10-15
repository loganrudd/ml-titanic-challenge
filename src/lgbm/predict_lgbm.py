import pandas as pd
import numpy as np
import pickle
import os


def predict(data):
    """
    Method to get predictions from PyFuncModel model
    arg: data: json object with processed features to make a prediction
    """

    localdir = os.path.abspath(os.path.dirname(__file__))

    # Load model as a PyFuncModel.
    loaded_model = pickle.load(open(os.path.join(localdir, "../../models/rf_clf.pkl"),
                                    "rb"))

    # Predict on a Pandas DataFrame.
    prediction = loaded_model.predict(pd.DataFrame(data))

    json_prediction = np.array2string(prediction)

    return json_prediction
