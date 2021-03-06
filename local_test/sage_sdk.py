import sagemaker as sage


def train_and_deploy():

    # AmazonSageMaker ExecutionRole
    role = "arn:aws:iam:"

    sess = sage.Session()
    account = sess.boto_session.client('sts').get_caller_identity()['Account']

    # path to the s3 location to store the models
    bucket_path = "s3://mdb-sagemaker/models/"
    region = sess.boto_session.region_name

    # location of the mindsdb container
    image = '{}.dkr.ecr.{}.amazonaws.com/mindsdb_lts:latest'.format(account, region)

    # note that to_predict is required in hyperparameters
    mindsdb_impl = sage.estimator.Estimator(image,
                        role, 1, 'ml.m4.xlarge',
                        output_path=bucket_path,
                        sagemaker_session=sess,
                        base_job_name="mindsdb-lts-sdk",
                        hyperparameters={"to_predict": "Class"})

    # read data from                     
    dataset_location = 's3://mdb-sagemaker/diabetes.csv'
    mindsdb_impl.fit(dataset_location)
    print('DONE')

    # deploy container
    predictor = mindsdb_impl.deploy(1, 'ml.m4.xlarge', endpoint_name='mindsdb-impl')

    with open('test_data/diabetes-test.csv', 'r') as reader:
        when_data = reader.read()
    result = predictor.predict(when_data).decode('utf-8')
    print("Result: ", result)
    return result


if __name__ == "__main__":
    train_and_deploy()