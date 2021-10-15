from flask import Flask, request, jsonify
from src.lgbm.predict_lgbm import predict
import numpy as np
import joblib
import pickle as p

# initialize app
app = Flask(__name__)

# route to URL with predict function
@app.route('/api', methods=['POST'])
def route_prediction():
    data = request.get_json()
    print(data)
    prediction = predict(data)
    return jsonify(prediction)


@app.errorhandler(404)
def page_not_found(e):
    return("This page does not exist yet or is currently under construction."
           " Check back soon! (404)")


@app.errorhandler(500)
def handle_internal_server_error(e):
    return("Sorry, there has been an internal server error.")


# run app on port 6000
if __name__ == '__main__':
    app.run()
