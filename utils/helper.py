from torch.utils.data import DataLoader, random_split
import numpy as np

from .custom_dataset import CustomDataset

import constants as c


def create_dataloaders():
    dataset = CustomDataset(c.PATH_TRAIN)
    train_split, valid_split = random_split(
        dataset,
        [c.TRAIN_SPLIT / (1 - c.TEST_SPLIT), c.VALID_SPLIT / (1 - c.TEST_SPLIT)],
    )
    train_dl = DataLoader(train_split, c.BATCH_SIZE)
    valid_dl = DataLoader(valid_split, c.BATCH_SIZE)
    return train_dl, valid_dl


def get_accuracy(predictions, labels):
    good_count = 0
    for predication, label in zip(predictions, labels):
        if np.argmax(predication) == np.argmax(label):
            good_count += 1
    accuracy = good_count / len(predictions)
    return accuracy
