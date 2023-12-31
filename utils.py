import numpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm



class LSTMNet(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int,
                 num_layers: int,
                 hidden_size: int,
                 bidirectional: bool,
                 out_features: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=2 * hidden_size if bidirectional else hidden_size, out_features=out_features)

    def forward(self, inputs):
        emb_X = self.emb(inputs)
        out = self.lstm(emb_X)[0]
        out = self.fc(out).permute(0, 2, 1)
        return out


def build_model(config: dict) -> torch.nn.Module:
    return LSTMNet(**config["model_params"])

def get_predictions(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[numpy.array, numpy.array]:
    y_true, y_pred = [], []
    model.to(device)
    with torch.no_grad():
        model.eval()
        for inputs, label in tqdm(test_dataloader):
            inputs, label = inputs.to(device), label.to(device)
            outputs = model.forward(inputs)
            y_true.extend(label.flatten().detach().cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().flatten().tolist())
    return numpy.array(y_true), numpy.array(y_pred)


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str='cpu') -> float:
    
    model.train()
    train_loss = 0

    for inputs, label in train_dataloader:
        inputs, label = inputs.to(device), label.to(device)
        outputs = model.forward(inputs)
        loss = criterion.forward(outputs, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return train_loss


def test(model: torch.nn.Module,
         test_dataloader: torch.utils.data.DataLoader,
         criterion: torch.nn.Module,
         task_type: str,
         metric: list=None,
         train_or_test_mode: str="test",
         device: str='cpu') -> tuple[float, list]:
    
    model.to(device=device)
    with torch.no_grad():

        if train_or_test_mode=='test':
            model.eval()

        val_loss = 0

        for inputs, label in test_dataloader:
            inputs, label = inputs.to(device), label.to(device)
            outputs = model.forward(inputs)
            val_loss += criterion(outputs, label).item()
            y_true = label.flatten().detach().cpu()
            if task_type=="clf":
                y_pred = outputs.argmax(dim=1).cpu().flatten()
            elif task_type=="reg":
                y_pred = outputs.detach().cpu().flatten()
            for m in metric:
                m.update(y_pred, y_true)
                
        return val_loss, list(map(lambda x: x.compute(), metric))