import torch
from torch import nn


# Define model: network h in the paper
class StringEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(StringEmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        # layer dimensions: 50 -> 300 -> 300 -> 300 (if embedding size is 300)
        self.fc1 = nn.Linear(50, embedding_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        hidden1 = self.fc1(x)
        tanh1 = self.tanh(hidden1)
        hidden2 = self.fc2(tanh1)
        tanh2 = self.tanh(hidden2)
        hidden3 = self.fc3(tanh2)
        output = self.tanh(hidden3)

        return output


# Define model: network g in the paper
class StringScalingModel(nn.Module):
    def __init__(self, embedding_size):
        super(StringScalingModel, self).__init__()
        self.embedding_size = embedding_size
        # layer dimensions: 50 -> 300 -> 300 -> 1 (if embedding_size is 300)
        self.fc1 = nn.Linear(50, embedding_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, 1)

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu(hidden2)
        hidden3 = self.fc3(relu2)
        output = torch.square(hidden3)

        return output

