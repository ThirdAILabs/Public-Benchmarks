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

The first step to launch a distributed training job is to initialize a Ray Cluster using the autoscalar_aws.yml config file. 
### Starting Ray Cluster on AWS

To set up a Ray cluster on AWS, please follow the instructions below:

1. Ensure that your AWS account has been set up by your AWS administrator.
2. Install the AWS SDK by running the following command: `pip install -U boto3`.
3. Configure your AWS credentials by executing the command `aws configure`. Fill in the `aws_access_key_id` and `aws_secret_access_key` values provided to you via email from AWS. You may leave other fields empty for minimal use cases.
4. Once the `boto3` setup is complete, you can run the following script to test if Ray is functioning correctly on AWS:

    ```bash
        # Download the example-full.yaml file
        wget https://raw.githubusercontent.com/ray-project/ray/master/python/ray/autoscaler/aws/example-full.yaml

        # Create or update the cluster. Once the command finishes, it will display
        # the command that can be used to SSH into the cluster head node.
        ray up example-full.yaml

        # Access the head node via a remote shell.
        ray attach example-full.yaml

        # Try running a Ray program.
        python -c 'import ray; ray.init()'
        exit

        # Terminate the cluster when your workload is complete.
        ray down example-full.yaml
    ```

5. Remember to terminate the machine once your workload has finished. 
6. Once you have completed the setup, execute the command `ray up aws_autoscaler.yaml` to start the cluster and begin running your workload.

Before moving to the next step, make sure the entire cluster is up before running a workload. To check whether the cluster is up, use one of the following commands:

```
ray exec <location-of-autoscaler-file-locally> 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
```

or

```
ray status
```

The first command is used to tail the logs and monitor the cluster's status. The second command provides a summary of the cluster's status on the head node.

Please ensure that you follow these instructions carefully to set up and utilize a Ray cluster on AWS.

Please refer to the official [Ray documentation](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html) for more details on how to set up a Ray Cluster on AWS via a yaml config. This yaml file will also install the required software libraries on all of the machines.


## Configure Cloud Storage for Ray

Starting in `Ray 2.7`, Ray AIR (Train and Tune) will require users to pass in a `cloud storage` or `NFS` path if running distributed training or tuning jobs. We have used `AWS S3` for the same. Refer to this [issue](https://github.com/ray-project/ray/issues/37177) for more information.
- Note : You can run the code without configuring cloud storage till version `2.6.1` 

```bash
import thirdai.distributed_bolt as dist
from ray.air.config import RunConfig

run_config = RunConfig(
    name="experiment_name",
    storage_path="s3://bucket-name/experiment_results",
)

# Use cloud storage in Train by configuring `RunConfig(storage_path)`.
trainer = dist.BoltTrainer(
    ...
    run_config=run_config
)
```

However you need to make sure all nodes in cluster have access to `AWS S3`. Launching cluster by default passes the credentials on launching machine(your local machine) to `head node` but `worker machines` launched from `head node` do not have those access. Refer to this [issue](https://github.com/ray-project/ray/issues/18186) for more information.

To solve the issue, pass a custom `aws instance-profile`(which has `S3` access) to the `worker node-config`.
```bash
ray.worker.default:
    node_config:
        InstanceType: c5.12xlarge
        ImageId: ami-02f3416038bdb17fb # Deep Learning AMI (Ubuntu) Version 30
        # By default, worker node launched from head doesn't have s3 access. 
        # Hence, we pass a custom instance-profile that grants the s3 access.
        IamInstanceProfile:
            Arn: arn:aws:iam::YOUR_AWS_ACCOUNT_NUMBER:YOUR_INSTANCE_PROFILE
```

## Training

To begin the training, follow these steps:

1. Once cluster setup is done, ssh to the head node using ssh-command mentioned under `Get a remote shell to the cluster manually` under `Useful commands`. After ssh-ing into the head-node. 
2. To proceed, first navigate to the `~/Public-Benchmarks/criteo_distributed_training` directory. Next, you have two options: 

    a. If you wish to run all the demos at once, use the `run.sh <activation-key>` script. This will ensure smooth execution and provide a comprehensive overview.
    Note: To obtain an activation key, please reach out to us via email at contact@thirdai.com.

    b. Alternatively, if you prefer to run a smaller training job to verify the setup before running all the demos simultaneously, you can utilize the `run_minimal.sh <activation-key>` script. This will help confirm that everything is properly configured for executing the complete set of demos in one go.


## Evaluation
After completing the training, the evaluation script will also run on the same node. If the node has limited resources, we recommend saving the model after training and running the evaluation on a separate node with more memory and RAM after loading the model.

## Stop Ray Cluster
Once the demo is done running, make sure to stop the ray cluster using `ray down aws_autoscalar.yaml`. 