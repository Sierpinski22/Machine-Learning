import torch
import torch.nn as nn

Loss_func = nn.MSELoss()


def build_net():
    net = nn.Sequential(nn.Linear(9, 5),
                        nn.ReLU(),
                        nn.Linear(5, 3),
                        nn.ReLU(),
                        nn.Linear(3, 1),
                        nn.Sigmoid())

    optimizer = torch.optim.Adam(net.parameters(), 0.002)
    return net, optimizer


def generate(neighbors, net):
    neighbors = torch.tensor(neighbors, dtype=torch.float32)
    new_state = net(neighbors)
    return new_state.item()


def train(neighbors, target, net, optimizer):
    neighbors = torch.tensor(neighbors, dtype=torch.float32)
    target = torch.tensor([target], dtype=torch.float32)

    new_state = net(neighbors)

    loss = Loss_func(new_state, target)
    loss.backward()


    optimizer.step()
    optimizer.zero_grad()

    return new_state.item()


def save(model):
    torch.save(model.state_dict(), 'param.pth')


def load():
    net, _ = build_net()
    net.load_state_dict(torch.load('param.pth'))
    return net
