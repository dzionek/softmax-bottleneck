import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def save_results(args, results):
    file_name = f'{args.save_dir}/{args.experiment_name}'
    while os.path.exists(file_name):
        file_name += '1'

    with open(file_name + '_log', 'w') as f:
        for i, log in enumerate(results['log']):
            f.write(f'For seed {i+1}/{args.num_seeds}\n')
            f.write('\n'.join(log))
            f.write('\n\n')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, args.epochs+1)
    ax1.plot(epochs, results['train_loss'][0], 'b', label='Training Loss', alpha=0.3)
    ax1.plot(epochs, results['test_loss'][0], 'r', label='Test Loss', alpha=0.3)
    for i in range(1, len(results['train_loss'])):
        ax1.plot(epochs, results['train_loss'][i], 'b', alpha=0.3)
        ax1.plot(epochs, results['test_loss'][i], 'r', alpha=0.3)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Average Loss')
    ax1.legend()

    ax2.plot(epochs, results['train_accuracy'][0], 'b', label='Training Accuracy', alpha=0.3)
    ax2.plot(epochs, results['test_accuracy'][0], 'r', label='Test Accuracy', alpha=0.3)
    for i in range(1, len(results['train_loss'])):
        ax2.plot(epochs, results['train_accuracy'][i], 'b', alpha=0.3)
        ax2.plot(epochs, results['test_accuracy'][i], 'r', alpha=0.3)
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.savefig(f'{file_name}.png', bbox_inches='tight')



    with open(file_name + '_scores', 'w') as f:
        for i in range(len(results['train_loss'])):
            results["train_loss"][i] = list(map(str, results["train_loss"][i]))
            results["test_loss"][i] = list(map(str, results["test_loss"][i]))
            results["train_accuracy"][i] = list(map(str, results["train_accuracy"][i]))
            results["test_accuracy"][i] = list(map(str, results["test_accuracy"][i]))
            results["rank"][i] = list(map(str, results["rank"][i]))
            f.write(f'Seed {i+1}/{args.num_seeds}\n')
            f.write(f'Train loss: {",".join(results["train_loss"][i])}\n')
            f.write(f'Test loss: {",".join(results["test_loss"][i])}\n')
            f.write(f'Train accuracy: {",".join(results["train_accuracy"][i])}\n')
            f.write(f'Test accuracy: {",".join(results["test_accuracy"][i])}\n')
            f.write(f'Rank: {",".join(results["rank"][i])}\n')