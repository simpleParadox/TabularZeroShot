import torch
from torch.nn import functional as F
import mlflow
import torch.nn as nn


class ErinModel(nn.Module):

    def __init__(self, in_size=768, hidden_size: int = 1, num_relations: int = 29, sequence_length:int = 50):
        super(ErinModel, self).__init__()

        # Just add one LSTM unit as the model followed by a fully connected layer and then a softmax.
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(sequence_length * hidden_size, num_relations)

    def forward(self, x):
        # First get the bert embeddings.
        # Then do the forward pass.
        # print("x size: ", x.size())
        x, (h_n, c_n) = self.lstm(x)
        # print("x size after lstm: ", x.size())
        x = torch.flatten(x, 1)
        # print("x flatten: ", x.size())
        x = self.fc(x)
        # print("x after linear layer: ", x.size())
        output = F.softmax(x, 1)
        return output

