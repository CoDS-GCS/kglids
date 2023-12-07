import torch
from torch import nn

EMBEDDING_SIZE = 300


# Define model: network h in the paper
class NumericalEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(NumericalEmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        # layer dimensions: 32 -> 300 -> 300 -> 300 (if embedding size is 300)
        self.fc1 = nn.Linear(32, embedding_size)
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


def load_numeric_embedding_model(path: str ):
    model = NumericalEmbeddingModel(EMBEDDING_SIZE)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

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



def load_string_embedding_model(path: str):
    model = StringEmbeddingModel(EMBEDDING_SIZE)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
    
    
# Define model: network h in the paper
class NaturalLanguageEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(NaturalLanguageEmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        # layer dimensions: 50 -> 300 -> 300 -> 300 (if embedding size is 50)
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
class NaturalLanguageScalingModel(nn.Module):
    def __init__(self, embedding_size):
        super(NaturalLanguageScalingModel, self).__init__()
        self.embedding_size = embedding_size
        # layer dimensions: 50 -> 300 -> 300 -> 1 (if embedding_size is 50)
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
    
def load_nl_embedding_model(path):
    model = NaturalLanguageEmbeddingModel(EMBEDDING_SIZE)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
    

