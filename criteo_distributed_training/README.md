## Introduction

This repository contains scripts for each training, including the main training script `train.py` and the dataset preprocessing file `process_to_csv.py`. Additionally, the `run.sh` script includes commands for running all the trainings, assuming the datasets are available.

We ran it on VMware vSphere cluster with 12, 24, and 48 nodes, each having 4 vCPUs and 8 GB RAM. The cluster consisted of 5 servers connected through interconnect with a modest communication speed of 10Gbps.

### Machine Configuration:

- **Head Node:**
  - Number of machines: 1
  - vCPUs: 4
  - RAM: 32 GB

- **Worker Nodes:**
  - Number of machines: 47
  - vCPUs: 4
  - RAM: 8 GB

Following is the evaluation of UDT on the **Click Through Prediction** Task.

| Parameters | Training Time | AUC    |
| ---------- | ------------- | ------ |
| 25M        | 69 min        | 0.7921 |
| 37.5M      | 96 min        | 0.7947 |
| 50M        | 128 min       | 0.7954 |

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

The first step to launch a distributed training job is to initialize a Ray Cluster using the autoscalar_aws.yml config file. Please refer to the official [Ray documentation](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html) for details on how to set up a Ray Cluster on AWS via a yaml config. This yaml file will also install the required software libraries on all of the machines.


## Training

To begin the training, follow these steps:

1. Once cluster setup is done, ssh to the head node using ssh-command mentioned under `Get a remote shell to the cluster manually` under `Useful commands`. After ssh-ing into the head-node. 
2. Then, move `~/Public-Benchmarks/criteo_distributed_training` either use `run.sh` to run all of the demo at once, or use train.py to run the experiments one-by-one.


## Evaluation
After completing the training, the evaluation script will also run on the same node. If the node has limited resources, we recommend saving the model after training and running the evaluation on a separate node with more memory and RAM after loading the model.
