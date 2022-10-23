import torch

EMBEDDING_SIZE = 300


def load_pretrained_model(model_class, model_path, embedding_size=EMBEDDING_SIZE):
    model = model_class(embedding_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
