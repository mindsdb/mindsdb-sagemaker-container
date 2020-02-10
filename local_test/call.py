import boto3
import argparse
import argcomplete


def load_data(dataset):
    '''
    Reads file and returns data
    '''
    with open(dataset, 'r') as reader:
        payload = reader.read()
    return payload


def call_endpoint(endpoint_name, data_location, data_type):
    '''
    Use Amazon SageMaker Runtime client to call the enpoint
    Returns result from the  endpoint
    '''
    runtime = boto3.client('runtime.sagemaker')
    payload = load_data(data_location)
    result = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload,
        ContentType=data_type
    )
    result = result['Body'].read().decode()
    print(result)


if __name__ == "__main__":
    '''
    Example call:
    python3 call.py --endpoint mindsdb-lts --dataset test_data/diabetes-test.json --content-type application/json
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint",
                        help='Add the name of the SageMaker endpoint.',
                        default='mindsdb-lts',
                        dest='endpoint',
                        required=True,
                        type=str)
    parser.add_argument("--dataset",
                        help='The location of the test dataset.',
                        required=True,
                        dest='data',
                        type=str)
    parser.add_argument("--content-type",
                        help='Mime type of the data, e.g text/csv',
                        required=False,
                        default='text/csv',
                        dest='type',
                        type=str)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    call_endpoint(args.endpoint, args.data, args.type)
