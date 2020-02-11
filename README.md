# MindsDB SageMaker Container
This repository contains the MindsDB containers for use with SageMaker.

MindsDB container supports two execution modes on SageMaker. Training, where MindsDB uses input data to train a new model and serving where it accepts HTTP requests and uses the previously trained model to do a prediction.

## Table of contents
 * [Build an image](#build-an-image)
 * [Test the container locally](#test-the-container-locally)
   * [Test directory](#test-directory)
   * [Run tests](#run-tests)
 * [Push the image to ECS](#push-the-image-to-amazon-elastic-container-service)
 * [Training](#training)
   * [Required parameters](#required-parameters)
   * [Add HyperParameters](#add-hyperparameters)
 * [Inference](#inference)
   * [Create model](#create-model)
   * [Create endpoint](#create-endpoint)
   * [Call endpoint](#call-endpoint)
 * [Using the SageMaker Python SDK](#using-the-sagemaker-python-sdk)
   * [Starting train job ](#starting-train-job)
   * [Deploy model and create endpoint](#deploy-model-and-create-endpoint)
   * [Delete the endpoint](#delete-the-endpoint)
 * [Other helpful resources](#other-usefull-resources)

## Build an image

Execute the following command to build the image:

```sh
docker build -t mindsdb-impl .
```

Note that `mindsdb-impl` will be the name of the image.

## Test the container locally

All of the files for testing the setup are located inside the local_test directory.

#### Test directory

* `train_local.sh`: Instantiate the container configured for training.
* `serve_local.sh`: Instantiate the container configured for serving.
* `predict.sh`: Run predictions against a locally instantiated server.
* `test-dir`:  This directory is mounted in the container.
*  `test_data`: This directory contains a few tabular format datasets used for getting the predictions.
* `input/data/training/file.csv`: The training data.
* `model`: The directory where mindsdb writes the model files.
* `output`: The directory where mindsdb can write its failure file.

All of the files under test-dir are mounted into the container and mimics the SageMaker directory structure.

#### Run tests
To train the model execute train script and specify the tag name of the docker image:
```sh
./train_local.sh mindsdb-impl
```
The train script will use the dataset that is located in the `input/data/training/` directory.

Then start the server:
```sh
./serve_local.sh mindsdb-impl
```

And make predictions by specifying the payload file in json format:
```sh
./predict.sh payload.json
```

## Push the image to Amazon Elastic Container Service

Use the shell script `build-and-push.sh`, to push the latest image to the Amazon Container Services.
You can run it as:
```sh
 ./build-and-push.sh mindsdb-impl 
```
The script will look for an AWS EC repository in the default region that you are using, and create a new one if that doesn't exist.

## Training 
When you create a training job, Amazon SageMaker sets up the environment, performs the training, then store the model artifacts in the location you specified when you created the training job.

### Required parameters
* **Algorithm source**: Choose `Your own algorithm` and provide  the registry path where the mindsdb image is stored in Amazon ECR  `846763053924.dkr.ecr.us-east-1.amazonaws.com/mindsdb_impl`
* **Input data configuration**: Choose S3 as a data source and provide path to the backet where the dataset is stored e.g
s3://bucket/path-to-your-data/
* **Output data configuration**: This would be the location where the model artifacts will be stored on s3 e.g
s3://bucket/path-to-write-models/

### Add HyperParameters
 You can use hyperparameters to finely control training. The required parameter for training models with mindsdb is:
 `to_predict` parameter. That is the column we want to learn to predict given all the data in the file e.g
 `to_predict = Class`

## Inference
You can also create a model, endpoint configuration and endpoint using [AWS Management Console ](https://console.aws.amazon.com/sagemaker/home). 

### Create model
Choose the role that has the AmazonSageMakerFullAccess IAM policy attached.
Next, you need to provide the location of the model artifacts and inference code.
* **Location of inference code image**: Location to the ECR image `846763053924.dkr.ecr.us-east-1.amazonaws.com/mindsdb_impl:latest`
* **Location of model artifacts - optional** Path to the s3 where the models are saved. This is the same location that you provide on train job `s3://bucket/path-to-write-models/`

### Create endpoint  
First, create an endpoint configuration. In the configuration, specify which models to deploy and hardware requirements for each. The required option is `Endpoint configuration name` and then add the previously created model. Then go to `Create and configure endpoint`, add the `Endpoint name`, and `Attach endpoint configuration`. Usually, it would take around few minutes to start the instance and create endpoint.

### Call endpoint
When the endpoint is in `InService` status, you can create python script or notebook from which you can get the predictions. 

```python
import boto3
# Set below parameters
bucket = 'mindsdb-sagemaker'
endpointName = 'mindsdb-impl'

params = '{"Plasma glucose concentration": 199, "Diastolic blood pressure": 84,"Age": 54'}
# Talk to SageMaker
client = boto3.client('sagemaker-runtime')
response = client.invoke_endpoint(
    EndpointName=endpointName,
    Body=params,
    ContentType='application/json',
    Accept='Accept'
)
print(response['Body'].read().decode('ascii'))
//mindsdb prediction response
{
"prediction": "* We are 96% confident the value of "Class" is positive.", 
 "model_confidence": [0.8310450414816538], 
 "class_confidence": [0.964147493532568]
}
```

## Using the SageMaker Python SDK
SageMaker provides Estimator implementation that runs SageMaker compatible custom Docker containers, enabling our own MindsDB implementation.

### Starting train job 
he Estimator defines how you can use the container to train. This is simple example that includes the required configuration to start training:

```python
import boto3
import re
import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role

role = get_execution_role()
account = sess.boto_session.client('sts').get_caller_identity()['Account']

sess = sage.Session()
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/mindsdb-impl:latest'.format(account, region)
mindsdb_impl = sage.estimator.Estimator(image,
                       role, 1, 'ml.m4.xlarge',
                       output_path="s3://{}/output".format(sess.default_bucket()),
                       sagemaker_session=sess)
dataset_location = 's3://bucket/path-to-your-data/'
mindsdb_impl.fit(dataset_location)
```
### Deploy model and create endpoint 
The model can be deployed to SageMaker by calling deploy method.
```python
predictor = mindsdb.deploy(1, 'ml.m4.xlarge', endpoint_name='mindsdb-impl')
```
The deploy method configures the Amazon SageMaker hosting services endpoint, deploy model and launches the endpoint to host the model. It returns RealTimePredictor object, from which you can get the predictions from.
```python
when = json.dumps({"Plasma glucose concentration": 162, "Diastolic blood pressure": 84,"Age": 54})
print(predictor.predict(when).decode('utf-8'))
```
The predict endpoint only accepts json data, so make sure to provide correct format.
### Delete the endpoint 
Don't forget to delete the endpoint when you are not using it.
```python
mindsdb.sagemaker_session.delete_endpoint('mindsdb-impl')
```

## Other usefull resources
 * [ Explainable AutoML with MindsDB](https://mindsdb.github.io/mindsdb/docs/basic-mindsdb)
 * [Getting started with Docker](https://docs.docker.com/get-started/)
 * [Amazon SageMaker examples](https://github.com/awslabs/amazon-sagemaker-examples)
    *  [Bring-your-own Algorithm Sample](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own/container)
 * [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)
