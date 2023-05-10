# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import time
import math
from tqdm import tqdm
import argparse


class CriteoBinDataset(Dataset):
    """Binary version of criteo dataset."""

    def __init__(
        self, data_file, batch_size=2048, max_ind_range=-1, bytes_per_feature=4
    ):
        # dataset
        self.tar_fea = 1  # single target
        self.den_fea = 13  # 13 dense  features
        self.spa_fea = 26  # 26 sparse features
        self.tad_fea = self.tar_fea + self.den_fea
        self.tot_fea = self.tad_fea + self.spa_fea

        self.batch_size = batch_size
        self.max_ind_range = max_ind_range
        self.bytes_per_entry = bytes_per_feature * self.tot_fea * batch_size

        self.num_entries = math.ceil(os.path.getsize(data_file) / self.bytes_per_entry)

        print("data file:", data_file, "number of batches:", self.num_entries)
        self.file = open(data_file, "rb")

        # hardcoded for now
        self.m_den = 13

    def __len__(self):
        return self.num_entries

    def __getitem__(self, idx):
        self.file.seek(idx * self.bytes_per_entry, 0)
        raw_data = self.file.read(self.bytes_per_entry)
        array = np.frombuffer(raw_data, dtype=np.uint32)
        return array

    def __del__(self):
        self.file.close()
