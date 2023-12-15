import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        self.val_loss_min = val_loss


def train_one_epoch(data_loader,model,criterion,optimizer,device):
    model.train()
    train_loss = 0.0
    for batch_idx, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Update train loss
        train_loss += loss.item() #? Funciona
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
        loss = criterion(target,labels)
        valid_loss += loss.item()
        # Accuracy return 
        _, predicted = torch.max(target.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return valid_loss,100*correct/total