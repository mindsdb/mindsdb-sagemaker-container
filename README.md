# mindsdb-sagemaker-container
This example shows how to package MindsDB for use with SageMaker.

SageMaker supports two execution modes: training where the algorithm uses input data to train a new model and serving where the algorithm accepts HTTP requests and uses the previously trained model to do an prediction.

## Build image

Execute the following command to build the image:

```sh
docker build -t mindsdb-sage .
```
Note that mindsdb-sage will be the name of the image.

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

## Push to Amazon EC2 Container Registry

## Model Creation

## Endpoint configuration

## Notebook
