from model import NeuralNetwork
from dataLoader import train_loader
from dataLoader import test_loader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch

def train_loop(dataloader, model, loss_fn, optimizer, batchsize):
    train_writer = SummaryWriter()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        t = min(batchsize, y.size(dim=0))
        logits = model(X)
        pred_probab = nn.Softmax(dim=1)(logits)
        target = torch.zeros(t, 4)
        for i in range(t):
            target[i][y[i]] = 1
        target = target.float()
        loss = loss_fn(pred_probab, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            train_writer.add_scalar('train_loss', loss, batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch % 200 == 0:
            torch.save(model.state_dict(), './weights/simple_model_weights')


def test_loop(dataloader, model, loss_fn,batchsize):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            logits = model(X)
            pred_probab = nn.Softmax(dim=1)(logits)
            t = min(batchsize, y.size(dim=0))
            target = torch.zeros(t, 4)
            for i in range(t):
                target[i][y[i]] = 1
            target = target.float()
            test_loss += loss_fn(pred_probab, target).item()
            correct += torch.sum(torch.eq(pred_probab.argmax(1), y)).item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

model=NeuralNetwork()
learning_rate = 1e-3
batch_size = 64
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer, train_loader.batch_size)
    test_loss = test_loop(test_loader, model, loss_fn, test_loader.batch_size)
    test_writer = SummaryWriter()
    test_writer.add_scalar('test_loss', test_loss, t)
print("Done!")