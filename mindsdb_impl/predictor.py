# This is the file that implements a flask server to do inferences. 

import os
import json
import flask
import pandas as pd

#Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
os.environ['MINDSDB_STORAGE_PATH'] = model_path

from mindsdb import Predictor

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
    # Get json data
    if flask.request.content_type == 'application/json':
        data = flask.request.json
        when = json.dumps(data)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(when))
    result = Predictor(name='mdbp').predict(when=json.loads(when))

    return flask.Response(response=str(result[0]), status=200, mimetype='application/json')