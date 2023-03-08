import math

import torch
from torch import nn
from torch.functional import F
from torchvision import datasets, transforms

EPSILON = 1e-8
NUM_CLASSES = 10

class SoftmaxNetwork(nn.Module):
    def __init__(self, d=128):
        super(SoftmaxNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, d)
        self.fc2 = nn.Linear(d, NUM_CLASSES)
        self.d = d

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.d > 10:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.d > 10:
            x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class MixtureOfSoftmaxesNetwork(nn.Module):
    def __init__(self, d=128, M=10):
        super(MixtureOfSoftmaxesNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, d)
        self.fc2 = nn.Linear(d, M*NUM_CLASSES)
        self.prior = nn.Parameter(torch.randn(M, 1), requires_grad=True)
        self.d = d
        self.M = M

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.d > 10:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.d > 10:
            x = self.dropout2(x)
        x = self.fc2(x)
        prior = F.softmax(self.prior, dim=1)
        x = F.softmax(x, dim=1)
        x = x.reshape([x.shape[0], NUM_CLASSES, self.M]) @ prior
        output = torch.log(x.squeeze() + EPSILON)
        return output


class SigSoftmaxNetwork(nn.Module):
    def __init__(self, d=128):
        super(SigSoftmaxNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, d)
        self.fc2 = nn.Linear(d, NUM_CLASSES)
        self.d = d

    def log_sigsoftmax(self, logits):
        stable_logits = logits - torch.max(logits)
        log_sigmoid = torch.log(torch.sigmoid(logits) + EPSILON)
        log_z = torch.log(
            torch.sum(torch.exp(stable_logits) * torch.sigmoid(logits), dim=-1,
                      keepdim=True) + EPSILON)
        return stable_logits + log_sigmoid - log_z

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.d > 10:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.d > 10:
            x = self.dropout2(x)
        x = self.fc2(x)
        output = self.log_sigsoftmax(x)
        return output

class MixtureOfSigSoftmaxesNetwork(nn.Module):
    def __init__(self, d=128, M=10):
        super(MixtureOfSigSoftmaxesNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, d)
        self.fc2 = nn.Linear(d, M*NUM_CLASSES)
        self.prior = nn.Parameter(torch.randn(M, 1), requires_grad=True)
        self.d = d
        self.M = M

    def sigsoftmax(self, logits):
        stable_logits = logits - torch.max(logits)
        unnormalized = torch.exp(stable_logits) * torch.sigmoid(logits)
        return unnormalized / torch.sum(unnormalized)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.d > 10:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.d > 10:
            x = self.dropout2(x)
        x = self.fc2(x)
        prior = F.softmax(self.prior, dim=1)
        x = self.sigsoftmax(x)
        x = x.reshape([x.shape[0], NUM_CLASSES, self.M]) @ prior
        output = torch.log(x.squeeze() + EPSILON)
        return output


class PlifNetwork(nn.Module):
    def __init__(self, d=128, K=100000, T=20, w_variance=1):
        super(PlifNetwork, self).__init__()
        self.K = K
        self.T = T
        self.w_variance = w_variance
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, d)
        self.fc2 = nn.Linear(d, NUM_CLASSES)
        self.plif_w = nn.Parameter(
            torch.randn(self.K) * self.w_variance + math.log(math.exp(1) - 1)
        )
        self.d = d

    def log_plif(self, logits):
        size = logits.size()
        logits = logits.view(-1)
        delta = 2. * self.T / self.K
        indices = torch.clamp(
            ((logits + self.T) / delta).detach().long(),
            max=self.K - 1, min=0
        )
        all_pos_w = nn.Softplus()(self.plif_w)
        all_pos_cumsum = torch.cumsum(all_pos_w, dim=-1) - all_pos_w
        pos_w = torch.gather(all_pos_w, -1, indices)
        # use gather, not take
        pos_w_cumsum = torch.gather(all_pos_cumsum, -1, indices)
        knots = (-self.T + delta * indices.float())
        knots = torch.tensor(
            knots, dtype=knots.dtype, device=logits.device
        )
        result = (logits - knots) * pos_w + delta * pos_w_cumsum
        return F.log_softmax(result.view(size), dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if self.d > 10:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.d > 10:
            x = self.dropout2(x)
        x = self.fc2(x)
        output = self.log_plif(x)
        return output

network = {
    'softmax': SoftmaxNetwork,
    'mos': MixtureOfSoftmaxesNetwork,
    'sigsoftmax': SigSoftmaxNetwork,
    'moss': MixtureOfSigSoftmaxesNetwork,
    'plif': PlifNetwork
}


def prepare_mnist(activation):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)

    return network[activation], dataset1, dataset2