# This is the file that implements a flask server to do inferences. 

import os
import json
import flask
import pandas as pd
from io import StringIO
import mindsdb
# Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the predictor was loaded correctly, if not throw error
    try:
        if not os.path.exists(model_path + '/mdbp_heavy_model_metadata.pickle'):
            raise IOError
        response = "Success"
        status = 200
    except Exception as e:
        response = str(e)
        status = 404
    return flask.Response(response=response, 
                          status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():

    # Avoid mindsdb storage path write access
    mindsdb.CONFIG.SAGEMAKER = 'True'
    mindsdb.CONFIG.MINDSDB_STORAGE_PATH = model_path
    # Get json data
    if flask.request.content_type == 'application/json':
        when = flask.request.json
        when_data = None
        print('Invoked with {} records'.format(when))
    elif flask.request.content_type == 'text/csv':
        req = flask.request.data.decode('utf-8')
        print(req)
        s = StringIO(req) 
        when = {}      
        when_data = pd.read_csv(s, header=0)
        print('Invoked with {} records'.format(when_data))
    else:
        return flask.Response(response='This predictor only supports CSV and JSON data', 
                              status=415, mimetype='text/plain')

    result = mindsdb.Predictor(name='mdbp').predict(when=when, when_data=when_data)

    mconfidence = [x['Class_model_confidence'] for x in result]
    cconfidence = [x['Class_confidence'] for x in result]
    response = {
        'prediction': str(result[0]),
        'model_confidence': mconfidence,
        'class_confidence': cconfidence
    }

    print('Response prediction: {}'.format(response['prediction']))
    return flask.Response(response=json.dumps(response), status=200, mimetype='application/json')
