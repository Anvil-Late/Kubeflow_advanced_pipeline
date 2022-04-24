FROM ubuntu:20.04
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip \
    && apt install -y python-is-python3
RUN pip3 --no-cache-dir install numpy scikit-learn pandas sagemaker boto3 scipy \
	seaborn matplotlib hyperopt xgboost

COPY utils .

