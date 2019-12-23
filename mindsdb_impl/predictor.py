# This is the file that implements a flask server to do inferences. 

import os
import json
from mindsdb import Predictor

import flask
import pandas as pd

#Define the path
prefix = '/opt/ml/'

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the predictor was loaded correctly
    try:
        from mindsdb import Predictor
        # mdb = Predictor(name='test_predictor')
        # mdb.get_model_data('test_predictor')
        status = 200
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():

    result = Predictor(name='home_rentals_price').predict(when={'number_of_rooms': 2,'number_of_bathrooms':1, 'sqft': 1190})

    print('The predicted price is ${price} with {conf} confidence'.format(price=result[0]['rental_price'], conf=result[0]['rental_price_confidence']))
    result = json.dumps(result)
    return flask.Response(response=result, status=200, mimetype='application/json')