import argparse
from statistics import mean
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import MNISTNet

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('Loss/train', mean(running_loss), epoch)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total
	
if __name__=='__main__':
    print(f'Using device: {device}')
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'mnist', help='experiment name')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    # Parameters
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    
    # Writer
    writer = SummaryWriter(f'runs/{exp_name}')
    
    # Transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = MNISTNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Training
    train(net, optimizer, trainloader, epochs=epochs)
    test_acc = test(net, testloader)
    print(f'Accuracy: {test_acc}')
    
    # Save model
    if not os.path.exists('weights'):
        os.makedirs('weights')
    torch.save(net.state_dict(), f'weights/{exp_name}.pth')
    
    # Add embeddings to tensorboard
    perm = torch.randperm(len(trainset.data))
    images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
    images = images.unsqueeze(1).float().to(device)
    with torch.no_grad():
        embeddings = net.get_features(images)
        writer.add_embedding(embeddings,
                    metadata=labels,
                    label_img=images, global_step=1)

    # save networks computational graph in tensorboard
    writer.add_graph(net, images)
    # save a dataset sample in tensorboard
    img_grid = torchvision.utils.make_grid(images[:64])
    writer.add_image('mnist_images', img_grid)
