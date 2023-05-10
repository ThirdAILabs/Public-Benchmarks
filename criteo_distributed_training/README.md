## Introduction

This repository contains scripts for each training, including the main training script `train.py` and the dataset preprocessing file `process_to_csv.py`. Additionally, the `run.sh` script includes commands for running all the trainings, assuming the datasets are available.

## Dataset

Preprocessed datasets for each training can be downloaded from the provided S3 links:
- Test Data: `aws s3 cp s3://thirdai-corp-public/test.txt . --no-sign-request`
- criteo_splitted_12 Data: `aws s3 cp s3://thirdai-corp-public/criteo_splitted_12/ . --recursive --no-sign-request`
- criteo_splitted_24 Data: `aws s3 cp s3://thirdai-corp-public/criteo_splitted_24/ . --recursive --no-sign-request`
- criteo_splitted_48 Data: `aws s3 cp s3://thirdai-corp-public/criteo_splitted_48/ . --recursive --no-sign-request`

However, if you need to re-preprocess the data from scratch, follow these instructions:

1. Download the binary dataset from the [Facebook Research DLRM repository](https://github.com/facebookresearch/dlrm/blob/main/data_loader_terabyte.py).
2. Run `process_to_csv.py` with the downloaded binary data. This will process the binary data into CSV files.
3. Use the processed CSV files as the training data.

## Training Setup

To set up the training, follow these steps:

1. Install `thirdai`.
2. Install `pygloo`. Note that PyPI wheels are broken, so you might need to build from source, or you can download wheels from [pratkpranav/pygloo release 0.2.0](https://github.com/pratkpranav/pygloo/releases/tag/0.2.0) according to your Python version.
3. Initialize a Ray cluster. Note that the `run.sh` script assumes the cluster already has 48*4 CPUs and each of the workers has access to their training data. If you are trying to run this on a different cluster, you can comment the training `run.sh` as per training.
   - Placing the preprocessed dataset there would be best if you have a shared mount among all workers. If that is not the case, one workaround we can do is to save each of the datasets with the same file name on each node, and they will be loaded independently. In this case, you might need to change the name of the train files too in `train.py`.

## Training

To begin the training, follow these steps:

1. Once the setup is complete and the training data is processed, each node should have access to its training data. If the training data is downloaded from S3 buckets, you can directly run `run.sh` after editing train and test files locations. Otherwise, as explained in the last section, you might need to change the training data location.
   - Note that `RayTrainingClusterConfig` in `train.py` is a wrapper that calls `ray.init()` inside it, so there is no need to call `ray.init()` initially. It also provides options to add `cluster_address` (if accessing a remote cluster) and `runtime_environment` (runtime environment), which are directly passed to `ray.init()` called inside `RayTrainingClusterConfig`.
2. All the parameters, except embedding dimensions (which varies the model size), are already set by default to an optimal value. Hence, to reproduce results, you might not need to change them.


## Evaluation
After completing the training, the evaluation script will also run on the same node. If the node has limited resources, we recommend saving the model after training and running the evaluation on a separate node with more memory and RAM after loading the model.