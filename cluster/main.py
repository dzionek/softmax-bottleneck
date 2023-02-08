import argparse

from mnist import prepare_mnist
from training import run_experiment
from stats import save_stats

parser = argparse.ArgumentParser(
    description='Runs experiments for the thesis')

parser.add_argument('experiment_name')
parser.add_argument('dataset', choices=['mnist'])
parser.add_argument('activation', choices=[
    'softmax', 'sigsoftmax', 'plif'])
parser.add_argument('-c', '--compute',
                    choices=['mps', 'cpu', 'cuda'], default='cpu')
parser.add_argument('-e', '--epochs', type=int, default=1)
parser.add_argument('-d', type=int, required=True)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-t', '--test_batch_size', type=int, default=10000)
parser.add_argument('-g', '--gamma', type=int, default=0.7)
parser.add_argument('-s', '--seed', type=int, default=1)
# parser.add_argument('--save_model', action='store_true')

args = parser.parse_args()

if args.dataset.lower() == 'mnist':
    experiment = prepare_mnist(args.activation)

stats = run_experiment(args, *experiment)
save_stats(args, stats)
