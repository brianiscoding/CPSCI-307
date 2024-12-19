from torch.utils.data import DataLoader
import torch

from model_src.model import Model
from model_src.utils import CustomDataset, get_accuracy


import constants as c


if __name__ == "__main__":
    dataset = CustomDataset(c.PATH_TEST)
    test = DataLoader(dataset, c.BATCH_SIZE)

    model = Model()
    model.load_state_dict(
        torch.load(f"{c.PATH_CHECKPOINTS}/{c.TEST_MODEL}.pth", weights_only=True)
    )
    model.eval()

    with torch.no_grad():
        print("item label prediction")
        for data in test:
            inputs, labels, items = data

            labels = torch.argmax(labels, dim=1)
            predictions = model(inputs)
            predictions = torch.argmax(predictions, dim=1)

            for item, label, prediction in zip(items, labels, predictions):
                label = c.CATEGORIES[label.item()]
                prediction = c.CATEGORIES[prediction.item()]
                print(item, label, prediction)
