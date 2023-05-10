import argparse
from thirdai import bolt, licensing
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import thirdai.distributed_bolt as d_bolt

licensing.activate("<YOUR LICENSE KEY HERE>")


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
        type=int,
        required=True,
        metavar="N",
        help="Number of nodes to use for distributed training of the UDT model",
    )
    parser.add_argument(
        "--cpus_per_node",
        type=int,
        default=4,
        metavar="N",
        help="Number of CPUs allocated per node for the distributed training (default: 4)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the test file",
    )
    parser.add_argument(
        "--training_folder",
        type=str,
        required=True,
        metavar="FOLDER",
        help="Path to the folder containing training files with integer names (0 to num_nodes-1)",
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
        default=100,
        metavar="N",
        help="Maximum number of in-memory batches (default: 100)",
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
        os.path.join(args.training_folder, f"train_file{file_id}.txt")
        if file_id >= 10
        else os.path.join(args.training_folder, f"train_file0{file_id}.txt")
        for file_id in range(NUM_NODES)
    ],
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    max_in_memory_batches=args.max_in_memory_batches,
)
en = time.time()
print("Training Time:", en - st)

tabular_model.save(filename="udt_click_prediction.model")


# This part of code is memory/CPU intensive as we would be loading the whole test data(10 GB) for evaluation.
# If the head machine doesn't have enough memory and RAM. It is recommended to run it on a separate machine.

tabular_model = bolt.UniversalDeepTransformer.load(
    filename="udt_click_prediction.model"
)


activations = tabular_model.evaluate(
    filename=args.test_file, metrics=["categorical_accuracy"]
)

true_labels = np.zeros(activations.shape[0], dtype=np.float32)
with open(args.test_file) as f:
    header = f.readline()
    count = 0
    for line in f:
        true_labels[count] = np.float32(line.split(",")[0])
        count += 1

roc_auc = roc_auc_score(true_labels, activations[:, 1])

print("ROC_AUC:", roc_auc)
