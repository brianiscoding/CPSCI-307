import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import csv

import constants as c


if __name__ == "__main__":
    data = {x1: {x2: 0 for x2 in c.CATEGORIES} for x1 in c.CATEGORIES}
    with open("predictions.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data[row[1]][row[2]] += 1

    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    for i, e in enumerate(data):
        axs[int(i / 3)][i % 3].pie(
            data[e].values(), labels=data[e].keys(), autopct="%1.1f%%"
        )
        axs[int(i / 3)][i % 3].set_title(f"{e} ({sum(data[e].values())})")
    plt.show()
