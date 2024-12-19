import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.utils.data import random_split
from csv import writer

import constants as c


def create_categories():
    categories = {}
    for category in c.CATEGORIES:
        if not os.path.isdir(f"{c.PATH_DATA}/{category}"):
            continue
        items = []
        for item in os.listdir(f"{c.PATH_DATA}/{category}"):
            if not os.path.isfile(f"{c.PATH_DATA}/{category}/{item}"):
                continue
            items.append(item[:-4])  # exclude ".csv"
        categories[category] = items
    return categories


def file_append(path, items, label):
    with open(path, "a") as file:
        write = writer(file)
        for item in items:
            write.writerow([item, label])


if __name__ == "__main__":
    # truncate both files
    with open(c.PATH_TRAIN, "w") as _:
        with open(c.PATH_TEST, "w") as _:
            pass

    print(f"category total train test")
    categories = create_categories()
    min_category_size = min([len(categories[x]) for x in categories])
    train_size = int((c.TRAIN_SPLIT + c.VALID_SPLIT) * min_category_size)
    for category in c.CATEGORIES:
        total_size = len(categories[category])
        test_size = total_size - train_size
        train, test = random_split(categories[category], [train_size, test_size])

        print(f"{category} {total_size} {len(train)} {len(test)}")
        file_append(c.PATH_TRAIN, train, category)
        file_append(c.PATH_TEST, test, category)
