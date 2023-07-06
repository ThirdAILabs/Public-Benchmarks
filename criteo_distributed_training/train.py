import argparse
import sys

from thirdai import bolt, licensing
import numpy as np
from sklearn.metrics import roc_auc_score
import thirdai.distributed_bolt as d_bolt

# Redirects stdout to a log file
path = "log.txt"
sys.stdout = open(path, "w")


licensing.activate("<YOUR LICENSE KEY HERE>")


def cpus_per_node_type(value):
    # 2 is here for testing purpose, as we might not want to start whole cluster
    valid_values = [2, 12, 24, 48]
    if int(value) not in valid_values:
        raise argparse.ArgumentTypeError(
            f"Invalid value for cpus_per_node: {value}. Valid values are {valid_values}"
        )
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed training with Ray for Universal Deep Transformer (UDT) model"
    )
    parser.add_argument(
        "--embedding_dimension",
        type=int,
        default=256,
        metavar="N",
        help="Embedding dimension for the UDT model (default: 256)",
    )
    parser.add_argument(
        "--num_nodes",
        type=cpus_per_node_type,
        required=True,
        default=2,
        metavar="N",
        help="Number of CPUs allocated per node for the distributed training. Valid values: 2, 12, 24, 48 (default: 12)",
    )
    parser.add_argument(
        "--cpus_per_node",
        type=int,
        default=4,
        metavar="N",
        help="Number of CPUs allocated per node for the distributed training (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="Number of epochs to train the model (default: 1)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        metavar="LR",
        help="Learning rate for the model (default: 0.005)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=720000,
        metavar="N",
        help="Batch size for training (default: 720000)",
    )
    parser.add_argument(
        "--max_in_memory_batches",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of in-memory batches (default: 10)",
    )
    args = parser.parse_args()
    return args


args = parse_args()

embedding_dimension = args.embedding_dimension
NUM_NODES = args.num_nodes
CPUS_PER_NODE = args.cpus_per_node


def ray_cluster_config(communication_type="gloo"):
    """
    This function initalizes RayTrainingCluster Config.

    RayTrainingClusterConfig:
        Initialize the class by connecting to an existing Ray cluster or creating a new one,
        starting Ray workers on each node, initializing logging, and creating
        Ray primary and replica worker configs. Computes and stores a number
        of useful fields, including num_workers, communication_type, logging,
        primary_worker_config, and replica_worker_configs.

        Args:
            num_workers (int): The number of workers in the Ray cluster.
            requested_cpus_per_node (int, optional): The number of requested CPUs per node. Defaults to num of cpus on the current node.
            communication_type (str, optional): The type of communication between workers. Defaults to "circular". Use "gloo" for reproducing the results.
            cluster_address (str, optional): The address to pass to ray.init() to connect to a cluster. Defaults to "auto".
            runtime_env (Dict, optional): Environment variables, package dependencies, working
                directory, and other dependencies a worker needs in its environment
                to run. See
                https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#:~:text=A%20runtime%20environment%20describes%20the,on%20the%20cluster%20at%20runtime
                Defaults to an empty dictionary.
            ignore_reinit_error (bool, optional): Whether to suppress the error that a cluster
                already exists when this method tries to create a Ray cluster. If
                this is true and a cluster exists, this constructor will just
                connect to that cluster. Defaults to False.
            log_dir (str, optional): The directory where the log files will be stored. Defaults to a temporary directory.

    """
    cluster_config = d_bolt.RayTrainingClusterConfig(
        num_workers=NUM_NODES,
        requested_cpus_per_node=CPUS_PER_NODE,
        communication_type=communication_type,
    )
    return cluster_config


def download_data_from_s3(s3_file_address, local_file_path):
    # Remove the "s3://" prefix
    trimmed_address = s3_file_address.replace("s3://", "")

    # Split the trimmed address into bucket name and object key
    split_address = trimmed_address.split("/", 1)
    object_key = split_address[1]

    s3_bucket_url = f"https://thirdai-corp-public.s3.us-east-2.amazonaws.com"

    file_url = f"{s3_bucket_url}/{object_key}"
    import urllib.request

    try:
        urllib.request.urlretrieve(file_url, local_file_path)
        print(f"File downloaded successfully: {local_file_path}")
    except urllib.error.URLError as e:
        raise RuntimeError("Error occurred during download:", e)


# TODO(pratik): Add file reading from s3 back once, we solve this issue(https://github.com/ThirdAILabs/Universe/issues/1487
def down_s3_data_callback(data_loader):
    s3_file_address = data_loader.train_file
    local_file_path = (
        "/home/ubuntu/train_file"  # The path where you want to save the downloaded file
    )

    download_data_from_s3(s3_file_address, local_file_path)
    data_loader.train_file = local_file_path


training_data_folder = "criteo-sample-split"
if args.num_nodes == 12:
    training_data_folder = "criteo-split-12"
if args.num_nodes == 24:
    training_data_folder = "criteo-split-24"
if args.num_nodes == 48:
    training_data_folder = "criteo-split-48"


data_types = {
    f"numeric_{i}": bolt.types.numerical(range=(0, 1500)) for i in range(1, 14)
}
data_types.update({f"cat_{i}": bolt.types.categorical() for i in range(1, 27)})
data_types["label"] = bolt.types.categorical()

tabular_model = bolt.UniversalDeepTransformer(
    data_types=data_types,
    target="label",
    n_target_classes=2,
    integer_target=True,
    options={"embedding_dimension": embedding_dimension},
)

import time


st = time.time()
tabular_model.train_distributed(
    cluster_config=ray_cluster_config(),
    filenames=[
        f"s3://thirdai-corp-public/{training_data_folder}/train_file{file_id}.txt"
        if file_id >= 10
        else f"s3://thirdai-corp-public/{training_data_folder}/train_file0{file_id}.txt"
        for file_id in range(NUM_NODES)
    ],
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    max_in_memory_batches=args.max_in_memory_batches,
    training_data_loader_callback=down_s3_data_callback,
)
en = time.time()
print("Training Time:", en - st)

tabular_model.save(filename="udt_click_prediction.model")


# This part of code is memory/CPU intensive as we would be loading the whole test data(11 GB) for evaluation.
# If the head machine doesn't have enough memory and RAM. It is recommended to run it on a separate machine.
tabular_model = bolt.UniversalDeepTransformer.load(
    filename="udt_click_prediction.model"
)

# Skip test for sample-training
if args.num_nodes == 2:
    print(
        "The demonstration run has been successfully completed. You may now proceed with the execution to complete the training process."
    )
else:
    # TODO(pratik): Add file reading from s3 back once, we solve this issue(https://github.com/ThirdAILabs/Universe/issues/1487)
    local_test_data = "/home/ubuntu/test_file"
    download_data_from_s3("s3://thirdai-corp-public/test.txt", local_test_data)

    from itertools import islice

    chunk_size = 1000000
    true_labels = []
    activations = []

    # define datatypes
    data_type_dict = [f"numeric_{i}" for i in range(1, 14)]
    data_type_dict.extend([f"cat_{i}" for i in range(1, 27)])

    with open(local_test_data) as f:
        header = f.readline()

        while True:
            test_sample_batch = []
            next_n_lines = list(islice(f, chunk_size))
            if not next_n_lines:
                break
            for line in next_n_lines:
                true_labels.append(np.float32(line.split(",")[0]))
                test_sample_batch.append(
                    dict(zip(data_type_dict, line.strip().split(",")[1:]))
                )

            activations.extend(tabular_model.predict_batch(test_sample_batch))

    true_labels = np.array(true_labels)
    activations = np.array(activations)
    roc_auc = roc_auc_score(true_labels, activations[:, 1])

    print("ROC_AUC:", roc_auc)
