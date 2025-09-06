import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=20,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss):

        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score > (self.best_score + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_one_epoch(data_loader, model, criterion, optimizer, device, half=False):
    model.train()
    train_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=half)

    for batch_idx, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        if half:
            with torch.cuda.amp.autocast():
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if half:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        train_loss += loss.item()
    return train_loss


@torch.no_grad()
def evaluate(data_loader, model, criterion, device):
    valid_loss = 0.0
    correct = 0
    total = 0

    # switch to evaluation mode
    model.eval()

    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)
        target = model(data)
        # Loss computing
        loss = criterion(target, labels)
        valid_loss += loss.item()
        # Accuracy return
        _, predicted = torch.max(target.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return valid_loss, 100 * correct / total


@torch.no_grad()
def get_latent_space(data_loader, model, device):
    X_latent = []
    Y_latent = []
    model.eval()

    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)
        embeddings = model(data)
        X_temp = embeddings.cpu().detach().numpy()
        Y_temp = labels.cpu().detach().numpy()
        for i in range(X_temp.shape[0]):
            X_latent.append(X_temp[i, :])
            Y_latent.append(Y_temp[i])

    return np.array(X_latent), np.array(Y_latent)
