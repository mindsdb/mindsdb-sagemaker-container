# This is the file that implements a flask server to do inferences.

import os
import json
import flask
import pandas as pd
from io import StringIO, BytesIO
import mindsdb
# Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')


def parse_data(content_type, data):
    '''
    Get the request content type and return data as DataFrame object
    '''
    excel_mime = ['application/vnd.ms-excel',
                  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']

    if content_type == 'application/json':
        req = data.json
        when_data = pd.DataFrame(req)
    elif content_type == 'text/csv':
        req = data.data.decode('utf-8')
        s = StringIO(req)
        when_data = pd.read_csv(s, header=0)
    elif content_type in excel_mime:
        req = data.data
        s = BytesIO(req)
        when_data = pd.read_excel(s, header=0)
    else:
        raise ValueError
    return when_data


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

    try:
        when_data = parse_data(flask.request.content_type, flask.request)
    except ValueError:
        return flask.Response(response='This predictor supports JSON,CSV and Excel data',
                              status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(when_data))
    result = mindsdb.Predictor(name='mdbp').predict(when_data=when_data)

    cconfidence = [x['Class_confidence'] for x in result]
    response = {
        'prediction': str(result[0]),
        'class_confidence': cconfidence[-1]
    }

    print('Response prediction: {}'.format(response['prediction']))
    return flask.Response(response=json.dumps(response),
                          status=200, mimetype='application/json')
