import torch
import torch.nn as nn

Loss_func = nn.MSELoss()


def build_net():
    net = nn.Sequential(nn.Linear(9, 7),
                        nn.ReLU(),
                        nn.Linear(7, 5),
                        nn.ReLU(),
                        nn.Linear(5, 4),
                        nn.ReLU(),
                        nn.Linear(4, 3),
                        nn.ReLU(),
                        nn.Linear(3, 1),
                        nn.Sigmoid()
                        )

    optimizer = torch.optim.Adam(net.parameters(), 0.002)

    return net, optimizer


def generate(neighbors, net):
    neighbors = torch.tensor(neighbors, dtype=torch.float32)
    new_state = net(neighbors)

    return new_state.item()


def train(neighbors, target, net, optimizer):  # div per 2
    neighbors = torch.tensor(neighbors, dtype=torch.float32)
    target = torch.tensor([target], dtype=torch.float32)

    new_state = net(neighbors)

    loss = Loss_func(new_state, target)
    loss.backward()
    # print(loss.item())

    optimizer.step()
    optimizer.zero_grad()

    return new_state.item()
