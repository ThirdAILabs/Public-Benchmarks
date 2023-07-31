import argparse
import sys
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import ray
from ray.air import session
from ray.train.torch import TorchConfig

from thirdai import bolt, licensing
import thirdai.distributed_bolt as dist
from utils import parse_args, setup_ray, get_udt_model, download_data_from_s3

# licensing.activate("<YOUR LICENSE KEY HERE>")


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
"""

NUM_NODES = args.num_nodes
CPUS_PER_NODE = args.cpus_per_node
EMBEDDING_DIM = args.embedding_dimension

training_data_folder = "criteo-sample-split"
if NUM_NODES == 12:
    training_data_folder = "criteo-split-12"
if NUM_NODES == 24:
    training_data_folder = "criteo-split-24"
if NUM_NODES == 48:
    training_data_folder = "criteo-split-48"


def train_loop_per_worker(config):
    model = get_udt_model(embedding_dimension=EMBEDDING_DIM)
    model = dist.prepare_model(model)

    train_file_s3 = f"s3://thirdai-corp-public/{training_data_folder}/train_file{session.get_world_rank():02}.txt"
    train_file_local = f"train_file{session.get_world_rank():02}.txt"
    download_data_from_s3(
        s3_file_address=train_file_s3, local_file_path=train_file_local
    )

    model.train_distributed_v2(
        filename=train_file_local,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_in_memory_batches=args.max_in_memory_batches,
    )
    pass

    session.report({}, checkpoint=dist.UDTCheckPoint.from_model(model))


st = time.time()

scaling_config = setup_ray(num_nodes=NUM_NODES, cpus_per_node=CPUS_PER_NODE)
trainer = dist.BoltTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={},
    scaling_config=scaling_config,
    backend_config=TorchConfig(backend="gloo"),
)
result_checkpoint_and_history = trainer.fit()
# ray.shutdown()

en = time.time()
print("Training Time:", en - st)

directory_path = "trained_models"
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

trained_model = result_checkpoint_and_history.checkpoint.get_model()
trained_model.save(
    filename=f"{directory_path}/udt_click_prediction_{NUM_NODES}_{EMBEDDING_DIM}.model"
)


# This part of code is memory/CPU intensive as we would be loading the whole test data(11 GB) for evaluation.
# If the head machine doesn't have enough memory and RAM. It is recommended to run it on a separate machine.
tabular_model = bolt.UniversalDeepTransformer.load(
    filename=f"{directory_path}/udt_click_prediction_{NUM_NODES}_{EMBEDDING_DIM}.model"
)

# Skip test for sample-training
if args.num_nodes == 2:
    print(
        "The demonstration run has been successfully completed. You may now proceed with the execution to complete the training process."
    )
else:
    print(
        f"Training is complete for model with {NUM_NODES} nodes and embedding dimension as {EMBEDDING_DIM}."
    )
    # TODO(pratik): Add file reading from s3 back once, we solve this issue(https://github.com/ThirdAILabs/Universe/issues/1487)
    local_test_data = "test_file.txt"
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
