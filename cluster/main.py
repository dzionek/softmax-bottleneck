import argparse
from collections import defaultdict
import os

from mnist import prepare_mnist
from training import run_experiment
from results import save_results
from seeds import seeds

parser = argparse.ArgumentParser(
    description='Runs experiments for the thesis')

parser.add_argument('experiment_name')
parser.add_argument('dataset', choices=['mnist'])
parser.add_argument('activation', choices=[
    'softmax', 'mos', 'sigsoftmax', 'moss', 'plif'])
parser.add_argument('-c', '--compute',
                    choices=['mps', 'cpu', 'cuda'], default='cpu')
parser.add_argument('-e', '--epochs', type=int, default=1)
parser.add_argument('-d', type=int, required=True)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-t', '--test_batch_size', type=int, default=10000)
# parser.add_argument('-g', '--gamma', type=int, default=0.7)
parser.add_argument('-s', '--num_seeds', type=int, default=1)
parser.add_argument('--save_dir', default='experiments')
# parser.add_argument('--save_model', action='store_true')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

experiment = None
if args.dataset.lower() == 'mnist':
    experiment = prepare_mnist(args.activation)

results = defaultdict(list)
for i, seed in enumerate(seeds[:args.num_seeds]):
    print(f'For seed {i+1}/{args.num_seeds}')
    stats = run_experiment(seed, args, *experiment)
    print('')
    for k, v in stats.items():
        results[k].append(v)

save_results(args, results)

