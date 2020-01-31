# MindsDB SageMaker Container
This example shows how to package MindsDB for use with SageMaker.

SageMaker supports two execution modes: training where the algorithm uses input data to train a new model and serving where the algorithm accepts HTTP requests and uses the previously trained model to do a prediction.

## Build image

Execute the following command to build the image:

```sh
docker build -t mindsdb-sage .
```

Note that `mindsdb-sage` will be the name of the image.

## Test the container locally

All of the files for testing the setup are located inside the local_test directory.

#### Test directory

* `train_local.sh`: Instantiate the container configured for training.
* `serve_local.sh`: Instantiate the container configured for serving.
* `predict.sh`: Run predictions against a locally instantiated server.
* `test-dir`:  This directory is mounted in the container.
* `payload.json`: Sample data for `when` clause that is used for predictions. Note that each key is the name of an input column and each value is the value for that cell in the column.
* `input/data/training/file.csv`: The training data.
* `model`: The directory where mindsdb writes the model files.
* `output`: The directory where mindsdb can write its success or failure file.

All of the files under test-dir are mounted into the container and mimics the SageMaker directory structure.

#### Run tests
To train the model execute train script and specify the tag name of the docker image:
```sh
./train_local.sh mindsdb-sage
```
Then start the server:
```sh
./serve_local.sh mindsdb-sage
```
And make predictions by adding the file with data that you want to make the predictions for:
```sh
./predict.sh payload.json
```

## Push the image to Amazon Elastic Container Service

Use the shell script `build-and-push.sh`, to push the latest image to the Amazon Container Services.
You can run it as:
```sh
 ./build-and-push.sh mindsdb-sage 
```
The script will look for an AWS EC repository in the default region that you are using, and create a new one if that doesn't exist.

## Training Jobs
When you create a training job, Amazon SageMaker sets up the environment, performs the training, then store the model artifacts in the location you specified when you created the training job.

### Required parameters
* **Algorithm source**: Choose `Your own algorithm` and provide  the registry path where the mindsdb image is stored in Amazon ECR  `846763053924.dkr.ecr.us-east-1.amazonaws.com/mindsdb_implementation`
* **Input data configuration**: Choose S3 as data source and provide path to the backet where the dataset is stored e.g
s3://bucket/path-to-your-data/
* **Output data configuration**: This would be the location where the model artifacts will be stored on s3 e.g
s3://bucket/path-to-write-models/

### Add HyperParameters
 You can use hyperparameters to finely control training. The required parameter for training models with mindsdb is:
 `to_predict` parameter. That is the column we want to learn to predict given all the data in the file e.g
 `to_predict = Class`
 
### Starting train job from code(using Estimator)
You can also use Estimator, an interface for SageMaker training. The Estimator defines how you can use the container to train. This is simple example that includes the required configuration to start training:

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
image = '{}.dkr.ecr.{}.amazonaws.com/mindsdb_implementation:latest'.format(account, region)
mindsdb_impl = sage.estimator.Estimator(image,
                       role, 1, 'ml.m4.xlarge',
                       output_path="s3://{}/output".format(sess.default_bucket()),
                       sagemaker_session=sess)
dataset_location = 's3://bucket/path-to-your-data/'
mindsdb_impl.fit(dataset_location)
```




## Model Creation

## Endpoint configuration

## Notebook
