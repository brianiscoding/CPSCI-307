import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from model_src.model import Model
from model_src.utils import create_dataloaders, get_accuracy

import constants as c


if __name__ == "__main__":
    wandb.init(project="Radar Classification")

    train_dl, valid_dl = create_dataloaders()
    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=c.LEARNING_RATE)

    def train_one_epoch():
        total_loss = 0.0
        for data in train_dl:
            inputs, labels, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    best_vloss = 1_000_000.0
    best_model = None

    for epoch in range(c.EPOCHS):
        model.train(True)
        avg_loss = train_one_epoch()
        total_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(valid_dl):
                vinputs, vlabels, _ = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                total_vloss += vloss
                accuracy = get_accuracy(voutputs, vlabels)

        wandb.log(
            {"train_loss": avg_loss, "val_loss": total_vloss, "accuracy": accuracy}
        )
        if total_vloss < best_vloss:
            best_vloss = total_vloss
            best_model = model.state_dict()

    torch.save(best_model, f"{c.PATH_CHECKPOINTS}/{best_vloss:.4f}.pth")
