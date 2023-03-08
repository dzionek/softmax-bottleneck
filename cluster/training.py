import time

import torch
from torch.functional import F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train(args, model, device, train_loader, optimizer, epoch, stats):
    stats['log'].append(f'Train Epoch: {epoch}/{args.epochs}')
    print(stats['log'][-1])
    start = time.time()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    end = time.time()
    stats['time'].append(round(end-start))
    stats['log'].append(f'Epoch training took {stats["time"][-1]}s.')
    print(stats['log'][-1])


def test(model, device, train_loader, test_loader, stats):
    model.eval()
    train_loss = 0
    train_correct = 0
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = F.nll_loss(output, target, reduction='sum').item()
            test_loss += batch_loss
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    test_loss /= len(test_loader.dataset)

    stats['train_loss'].append(round(train_loss, 4))
    stats['train_accuracy'].append(round(100.*train_correct / len(train_loader.dataset), 1))
    stats['log'].append(
        f'Training set: Average loss: {stats["train_loss"][-1]}, '
        f'Accuracy: {train_correct}/{len(train_loader.dataset)} ({stats["train_accuracy"][-1]}%)')
    print(stats['log'][-1])

    stats['test_loss'].append(round(test_loss, 4))
    stats['test_accuracy'].append(round(100.*test_correct / len(test_loader.dataset), 1))
    stats['log'].append(
        f'Test set: Average loss: {stats["test_loss"][-1]}, '
        f'Accuracy: {test_correct}/{len(test_loader.dataset)} ({stats["test_accuracy"][-1]}%)')
    print(stats['log'][-1])


def run_experiment(args, network, dataset1, dataset2):
    stats = {
        'log': [str(vars(args))], 'train_loss': [], 'train_accuracy': [],
        'test_loss': [], 'test_accuracy': [], 'time': []}
    torch.manual_seed(args.seed)
    device = torch.device(args.compute)
    stats['log'].append(f'Training a model on {args.dataset} using {device.type}'
                        f' and {args.activation} activation.')
    print(stats['log'][-1])

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = network(d=args.d).to(device)

    # stats['log'].append(f'The model has {sum(p.numel() for p in model.parameters())} parameters')
    # print(stats['log'][-1])

    optimizer = optim.Adam(model.parameters())

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, stats)
        test(model, device, train_loader, test_loader, stats)
        # scheduler.step()

    stats['log'].append(f'Done! Avg time per epoch {round(sum(stats["time"]) / args.epochs)}s.')
    stats['log'].append(f'Done! Total time {round(sum(stats["time"])/60)} min.')
    print(stats['log'][-1])

    return stats