import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CrossEntropyClassification(pl.LightningModule):
    def __init__(
        self,
        *layer_widths,
        activation=None,
        learning_rate=3e-4,
        weight_decay=1e-6,
    ):
        super().__init__()
        self.activation = nn.ReLU() if activation is None else activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # we have to store each layer in a ModuleList so that
        # pytorch can find the parameters of the model
        # when we pass model.parameters() to an optimizer
        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, x):
        # sequentially pass the output of each layer to the next layer
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # don't apply the activation to the last layer
        x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        # cross_entropy loss automatically applies softmax to the last layer
        # this converts the logits to probabilities
        loss = F.cross_entropy(logits, y)

        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)

        loss = F.cross_entropy(logits, y)

        self.log("validation_loss", loss)
        return loss
