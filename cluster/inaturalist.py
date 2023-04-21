import math

import torch
from torch import nn
from torch.functional import F
# from torch.utils.data import Subset
# from torchvision import datasets, transforms
# from transformers import ViTForImageClassification

EPSILON = 1e-8
NUM_CLASSES = 100


class SoftmaxNetwork(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc1 = nn.Linear(768, d)
        self.fc2 = nn.Linear(d, NUM_CLASSES)
        self.d = d

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class MixtureOfSoftmaxesNetwork(nn.Module):
    def __init__(self, d=64, M=10):
        super().__init__()
        self.fc1 = nn.Linear(768, d * M)
        self.fc2 = nn.Linear(d, NUM_CLASSES)
        self.d = d

        self.prior = nn.Parameter(torch.randn(M), requires_grad=True)
        self.d = d
        self.M = M

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = x.reshape((x.shape[0], self.M, self.d))

        prior = F.softmax(self.prior, dim=-1)
        output = F.softmax(self.fc2(x), dim=2)
        output = output * prior.view((1, self.M, 1)).repeat((x.shape[0], 1, NUM_CLASSES))
        output = torch.sum(output, dim=1)
        output = torch.log(output + EPSILON)
        return output


class SigSoftmaxNetwork(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.fc1 = nn.Linear(768, d)
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
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        output = self.log_sigsoftmax(x)
        return output


class MixtureOfSigSoftmaxesNetwork(nn.Module):
    def __init__(self, d=64, M=10):
        super().__init__()
        self.fc1 = nn.Linear(768, d * M)
        self.fc2 = nn.Linear(d, NUM_CLASSES)
        self.prior = nn.Parameter(torch.randn(M), requires_grad=True)
        self.d = d
        self.M = M

    def sigsoftmax(self, logits):
        stable_logits = logits - torch.max(logits)
        unnormalized = torch.exp(stable_logits) * torch.sigmoid(logits)
        return unnormalized / (
                    torch.sum(unnormalized, dim=1, keepdim=True) + EPSILON)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = x.reshape((x.shape[0], self.M, self.d))

        prior = F.softmax(self.prior, dim=-1)
        output = F.softmax(self.fc2(x), dim=2)
        output = output * prior.view((1, self.M, 1)).repeat((x.shape[0], 1, NUM_CLASSES))
        output = torch.sum(output, dim=1)
        output = torch.log(output + EPSILON)
        return output


class PlifNetwork(nn.Module):
    def __init__(self, d=64, K=100000, T=20, w_variance=1):
        super().__init__()
        self.fc1 = nn.Linear(768, d)
        self.fc2 = nn.Linear(d, NUM_CLASSES)

        self.d = d
        self.K = K
        self.T = T
        self.w_variance = w_variance
        self.plif_w = nn.Parameter(
            torch.randn(self.K) * self.w_variance + math.log(math.exp(1) - 1)
        )

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
        x = self.fc1(x)
        x = F.gelu(x)
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


# def prepare_dataset():
#     model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
#     model.classifier = nn.Identity()
#
#     processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
#     def my_normalization(x):
#         return model(processor(x, return_tensors='pt')['pixel_values']).logits.squeeze()
#
#     transform = transforms.Compose([transforms.Lambda(my_normalization)])
#
#     train = Subset(datasets.INaturalist('../data', version='2021_train_mini',
#                                         # download=True,
#                                         transform=transform
#                                         ),
#                    list(range(50 * NUM_CLASSES)))
#
#     test = Subset(datasets.INaturalist('../data', version='2021_valid',
#                                        # download=True,
#                                        transform=transform
#                                        ),
#                   list(range(10 * NUM_CLASSES)))
#
#     train_list = [(train[i][0].detach().cpu().numpy(), train[i][1]) for i in
#                  range(len(train))]
#
#     test_list = [(test[i][0].detach().cpu().numpy(), test[i][1]) for i in
#                  range(len(test))]
#
#     torch.save(train_list, f'inaturalist{NUM_CLASSES}_train.pt')
#     torch.save(test_list, f'inaturalist{NUM_CLASSES}_test.pt')

def prepare_inat(activation):
    # transform = lambda x: feature_extractor(x, return_tensors='pt')[
    #     'pixel_values']
    train = torch.load(f'inaturalist{NUM_CLASSES}_train.pt')
    test = torch.load(f'inaturalist{NUM_CLASSES}_test.pt')
    # train.dataset.transform = transform
    # test.dataset.transform = transform
    # train = Subset(datasets.INaturalist('../data', version='2021_train_mini',
    #                                     # download=True,
    #                                     transform=transform
    #                                     ),
    #                list(range(50 * NUM_CLASSES)))
    # test = Subset(datasets.INaturalist('../data', version='2021_valid',
    #                                    # download=True,
    #                                    transform=transform
    #                                    ),
    #               list(range(10 * NUM_CLASSES)))


    return network[activation], train, test
