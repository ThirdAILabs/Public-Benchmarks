import argparse
import os
import ray

from thirdai import bolt


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
    parser.add_argument(
        "--trainer_resources",
        type=int,
        default=16,
        metavar="N",
        help="Maximum number of in-memory batches (default: 10)",
    )
    parser.add_argument(
        "--activation_key",
        type=str,
        help="Activation Key for using ThirdAI",
    )
    args = parser.parse_args()
    return args


def setup_ray(num_nodes=2, cpus_per_node=4, trainer_resources=16):
    working_dir = os.path.dirname(os.path.realpath(__file__))

    ray.init(
        runtime_env={
            "working_dir": working_dir,
            "env_vars": {
                "OMP_NUM_THREADS": f"{cpus_per_node}",
                "GLOO_SOCKET_IFNAME": "ens5",
            },
            "excludes": ["trained_models", "*.txt"],
        },
        ignore_reinit_error=True,
    )
    scaling_config = ray.air.ScalingConfig(
        num_workers=num_nodes,
        use_gpu=False,
        trainer_resources={"CPU": trainer_resources},
        resources_per_worker={"CPU": cpus_per_node},
        placement_strategy="PACK",
    )
    return scaling_config


def get_udt_model(embedding_dimension=256):
    data_types = {
        f"numeric_{i}": bolt.types.numerical(range=(0, 1500)) for i in range(1, 14)
    }
    data_types.update({f"cat_{i}": bolt.types.categorical() for i in range(1, 27)})
    data_types["label"] = bolt.types.categorical()

    model = bolt.UniversalDeepTransformer(
        data_types=data_types,
        target="label",
        n_target_classes=2,
        integer_target=True,
        options={"embedding_dimension": embedding_dimension},
    )

    return model


# Streaming data from s3 in distrbuted training is slow. (Issue: https://github.com/ThirdAILabs/Universe/issues/1487)
# Hence we download train and test files.
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
