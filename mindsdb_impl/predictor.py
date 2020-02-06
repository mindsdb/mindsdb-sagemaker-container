# This is the file that implements a flask server to do inferences. 

import os
import json
import flask

# Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

os.environ['MINDSDB_STORAGE_PATH'] = model_path

from mindsdb import Predictor
# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the predictor was loaded correctly, if not throw error
    try:
        mdb = Predictor(name='mdbp')
        mdb.get_model_data('mdbp')
        response = 'Successfully loaded'
        status = 200
    except Exception as e:
        response = str(e)
        status = 404
    return flask.Response(response=response, 
                          status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():

    # Get json data
    if flask.request.content_type == 'application/json':
        when = flask.request.json
    else:
        return flask.Response(response='This predictor only supports JSON data', 
                              status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(when))
    result = Predictor(name='mdbp').predict(when=when)

    mconfidence = [x['Class_model_confidence'] for x in result]
    cconfidence = [x['Class_confidence'] for x in result]
    response = {
        'prediction': str(result[0]),
        'model_confidence': mconfidence,
        'class_confidence': cconfidence
    }

    print('Response prediction: {}'.format(response['prediction']))
    return flask.Response(response=json.dumps(response), status=200, mimetype='application/json')