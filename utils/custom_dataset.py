import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import constants as c


def create_categories_as_tensor():
    categories_as_tensor = {}
    for i, category in enumerate(c.CATEGORIES):
        category_as_tensor = torch.FloatTensor(
            [0 if i != j else 1 for j in range(len(c.CATEGORIES))]
        )
        categories_as_tensor[category] = category_as_tensor
    return categories_as_tensor


class CustomDataset(Dataset):
    def __init__(self, path_data):
        # one hot encode categories
        self.categories_as_tensor = create_categories_as_tensor()
        df = pd.read_csv(path_data, header=None)
        self.items = [str(x) for x in df[0].tolist()]
        self.labels = [str(x) for x in df[1].tolist()]
        self.len = len(df)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        item = self.items[index]
        label = self.labels[index]
        path_item = f"{c.PATH_DATA}/{label}/{item}.csv"
        # shape is (1, 312, 16)
        item_as_tensor = torch.FloatTensor(
            np.genfromtxt(path_item, delimiter=",")
        ).unsqueeze(0)
        label_as_tensor = self.categories_as_tensor[label]
        return item_as_tensor, label_as_tensor, item
