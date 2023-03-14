import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def save_results(args, results):
    if not os.path.exists('experiments'):
        os.mkdir('experiments')

    file_name = 'experiments/' + args.experiment_name
    while os.path.exists(file_name):
        file_name += '1'

    with open(file_name + '_log', 'w') as f:
        for i, log in enumerate(results['log']):
            f.write(f'For seed {i+1}/{args.num_seeds}\n')
            f.write('\n'.join(log))
            f.write('\n\n')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, args.epochs+1)
    ax1.plot(epochs, results['train_loss'][0], 'b', label='Training Loss')
    ax1.plot(epochs, results['test_loss'][0], 'r', label='Test Loss')
    for i in range(1, len(results['train_loss'])):
        ax1.plot(epochs, results['train_loss'][i], 'b')
        ax1.plot(epochs, results['test_loss'][i], 'r')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Average Loss')
    ax1.legend()

    ax2.plot(epochs, results['train_accuracy'][0], 'b', label='Training Accuracy')
    ax2.plot(epochs, results['test_accuracy'][0], 'r', label='Test Accuracy')
    for i in range(1, len(results['train_loss'])):
        ax2.plot(epochs, results['train_accuracy'][i], 'b')
        ax2.plot(epochs, results['test_accuracy'][i], 'r')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.savefig(f'{file_name}.png')



    with open(file_name + '_scores', 'w') as f:
        for i in range(len(results['train_loss'])):
            results["train_loss"][i] = list(map(str, results["train_loss"][i]))
            results["test_loss"][i] = list(map(str, results["test_loss"][i]))
            results["train_accuracy"][i] = list(map(str, results["train_accuracy"][i]))
            results["test_accuracy"][i] = list(map(str, results["test_accuracy"][i]))
            f.write(f'Seed {i+1}/{args.num_seeds}\n')
            f.write(f'Train loss: {",".join(results["train_loss"][i])}\n')
            f.write(f'Test loss: {",".join(results["test_loss"][i])}\n')
            f.write(f'Train accuracy: {",".join(results["train_accuracy"][i])}\n')
            f.write(f'Test accuracy: {",".join(results["test_accuracy"][i])}\n')