import os

import matplotlib.pyplot as plt

def save_stats(args, stats):
    if not os.path.exists('experiments'):
        os.mkdir('experiments')

    file_name = 'experiments/' + args.experiment_name
    while os.path.exists(file_name):
        file_name += '1'

    f = open(file_name, 'w')
    f.write('\n'.join(stats['log']))
    f.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, args.epochs+1)
    ax1.plot(epochs, stats['train_loss'], 'b', label='Training Loss')
    ax1.plot(epochs, stats['test_loss'], 'r', label='Test Loss')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Average Loss')
    ax1.legend()

    ax2.plot(epochs, stats['train_accuracy'], 'b', label='Training Accuracy')
    ax2.plot(epochs, stats['test_accuracy'], 'r', label='Test Accuracy')
    ax2.set_xlabel('Number of Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.savefig(f'{file_name}.png')