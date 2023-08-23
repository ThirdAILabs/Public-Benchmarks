import argparse
import sys
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import ray
from ray.air import session, RunConfig
from ray.train.torch import TorchConfig
from ray.tune import SyncConfig

import thirdai
from thirdai import bolt, licensing
import thirdai.distributed_bolt as dist
from utils import parse_args, setup_ray, get_udt_model, download_data_from_s3


args = parse_args()
"""
    The parameters returned by args :
        --embedding_dimension
        --num_nodes
        --cpus_per_node
        --epochs
        --learning_rate
        --batch_size
        --max_in_memory_batches
        --activation_key
"""

NUM_NODES = args.num_nodes
CPUS_PER_NODE = args.cpus_per_node
EMBEDDING_DIM = args.embedding_dimension
activation_key = args.activation_key
trainer_resources = args.trainer_resources
licensing.activate(activation_key)


training_data_folder = "criteo-sample-split"
if NUM_NODES == 12:
    training_data_folder = "criteo-split-12"
if NUM_NODES == 24:
    training_data_folder = "criteo-split-24"
if NUM_NODES == 48:
    training_data_folder = "criteo-split-48"


def train_loop_per_worker(config):
    licensing.activate(activation_key)

    thirdai.logging.setup(log_to_stderr=False, path="log.txt", level="info")

    model = get_udt_model(embedding_dimension=EMBEDDING_DIM)
    model = dist.prepare_model(model)

    train_file_s3 = f"s3://thirdai-corp-public/{training_data_folder}/train_file{session.get_world_rank():02}.txt"
    train_file_local = os.path.join(
        config.get("curr_dir"), f"train_file{session.get_world_rank():02}.txt"
    )
    download_data_from_s3(
        s3_file_address=train_file_s3, local_file_path=train_file_local
    )

    model.train_distributed_v2(
        filename=train_file_local,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size // args.num_nodes,
        max_in_memory_batches=args.max_in_memory_batches,
    )

    session.report({}, checkpoint=dist.UDTCheckPoint.from_model(model))


st = time.time()

scaling_config = setup_ray(
    num_nodes=NUM_NODES,
    cpus_per_node=CPUS_PER_NODE,
    trainer_resources=trainer_resources,
)

# Syncing files to the head node to be removed in Ray 2.7 in favor of cloud storage/NFS
# Hence we use s3 storage for future compatibility. (https://docs.ray.io/en/master/tune/tutorials/tune-storage.html#configuring-tune-with-a-network-filesystem-nfs)
# uncomment it if using S3 for checkpoint
# run_config = RunConfig(
# name=f"criteo_node_{NUM_NODES}_dim_{EMBEDDING_DIM}",
# storage_path="<Your S3 Bucket>/ThirdAI-Public-Benchmarks/",
# sync_config=SyncConfig(sync_artifacts=False, sync_period=1800),
# )

trainer = dist.BoltTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={"curr_dir": os.path.expanduser("~")},
    scaling_config=scaling_config,
    # run_config=run_config, # Note: uncomment it if using S3 for checkpoint
    backend_config=TorchConfig(backend="gloo"),
)
result_checkpoint_and_history = trainer.fit()

en = time.time()
print("Training Time:", en - st)

directory_path = os.path.join(os.getcwd(), "trained_models")
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

trained_model = result_checkpoint_and_history.checkpoint.get_model()
trained_model.save(
    filename=f"{directory_path}/udt_click_prediction_{NUM_NODES}_{EMBEDDING_DIM}.model"
)


# This part of code is memory/CPU intensive as we would be loading the whole test data(11 GB) for evaluation.
# If the head machine doesn't have enough memory and RAM. It is recommended to run it on a separate machine.

# define datatypes
data_type_dict = [f"numeric_{i}" for i in range(1, 14)]
data_type_dict.extend([f"cat_{i}" for i in range(1, 27)])


@ray.remote(num_cpus=trainer_resources / 2)
def eval_batch(filename, batch):
    licensing.deactivate()
    licensing.activate(activation_key)

    tabular_model = bolt.UniversalDeepTransformer.load(filename=filename)

    true_labels = []
    test_sample_batch = []
    for line in batch:
        true_labels.append(np.float32(line.split(",")[0]))
        test_sample_batch.append(dict(zip(data_type_dict, line.strip().split(",")[1:])))

    activations = tabular_model.predict_batch(test_sample_batch)[:, 1]

    return np.stack((true_labels, activations), axis=0)


# Skip test for sample-training
if args.num_nodes == 2:
    print(
        "The demonstration run has been successfully completed. You may now proceed with the execution to complete the training process."
    )
else:
    print(
        f"Training is complete for model with {NUM_NODES} nodes and embedding dimension as {EMBEDDING_DIM}."
    )

    local_test_data = "test_file.txt"
    download_data_from_s3("s3://thirdai-corp-public/test.txt", local_test_data)

    from itertools import islice

    chunk_size = 1000000
    outputs = []

    # define datatypes
    data_type_dict = [f"numeric_{i}" for i in range(1, 14)]
    data_type_dict.extend([f"cat_{i}" for i in range(1, 27)])

    model_path = (
        f"{directory_path}/udt_click_prediction_{NUM_NODES}_{EMBEDDING_DIM}.model"
    )

    with open(local_test_data) as f:
        header = f.readline()
        while True:
            test_sample_batch = []
            next_n_lines = list(islice(f, chunk_size))
            if not next_n_lines:
                break
            # Only run the task on the local node.
            task_ref = eval_batch.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(model_path, next_n_lines)
            outputs.append(task_ref)

    outputs = ray.get(outputs)
    merged_output = np.concatenate(outputs, axis=1)
    roc_auc = roc_auc_score(merged_output[0, :], merged_output[1, :])

    print("ROC_AUC:", roc_auc)
