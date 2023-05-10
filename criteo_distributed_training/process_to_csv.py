import numpy as np
import time
from data_loader import CriteoBinDataset

TRAIN_DATA = "location/terabyte_processed_train.bin"
TEST_DATA = "location/terabyte_processed_test.bin"
DESTINATION = "destination/"
NUM_FILES = 48


def save_data(filename, data):
    header = "label," + ",".join([f"numeric_{i}" for i in range(1, 14)]) + ","
    header += ",".join([f"cat_{i}" for i in range(1, 27)])
    np.savetxt(filename, data, delimiter=",", fmt="%.0f", header=header)


def process_data(train_dataset, i, max_numerical_column):
    big_batch = train_dataset[i]
    big_batch = np.reshape(big_batch, [-1, 40])
    y_train = big_batch[:, 0]

    indices = np.where(y_train == 0)[0]
    indices = np.random.choice(indices, int(0.2 * len(indices)), replace=False)
    indices = np.concatenate([indices, np.where(y_train == 1)[0]])
    np.random.shuffle(indices)
    big_batch = big_batch[indices]

    X_int_train = np.round(np.log(big_batch[:, 1:14] + 1), 2) * 100
    big_batch[:, 1:14] = X_int_train
    max_numerical_column = max(np.max(X_int_train), max_numerical_column)

    save_data(f"{DESTINATION}{i}.txt", big_batch)
    return max_numerical_column


train_block_size = 4200000000 // NUM_FILES
train_dataset = CriteoBinDataset(
    data_file=TRAIN_DATA,
    batch_size=train_block_size,
)

max_numerical_column = 0
t1 = time.time()
for i in range(NUM_FILES):
    max_numerical_column = process_data(train_dataset, i, max_numerical_column)
    t2 = time.time()
    print(t2 - t1)
    t1 = t2

test_block_size = 100_000_000
test_dataset = CriteoBinDataset(
    data_file=TEST_DATA,
    batch_size=test_block_size,
)

big_batch = test_dataset[0]
big_batch = np.array(big_batch)
big_batch = np.reshape(big_batch, [-1, 40])

X_int_train = np.round(np.log(big_batch[:, 1:14] + 1), 2) * 100
big_batch[:, 1:14] = X_int_train

save_data("./test.txt", big_batch)
