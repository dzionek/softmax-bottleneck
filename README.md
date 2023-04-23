# Alleviating the Low-Rank Softmax Bottleneck

The code complementing my undergraduate thesis "Alleviating the Low-Rank Softmax Bottleneck".

## How to reproduce results
1. Load [conda](https://docs.conda.io/en/latest/) environment or install libraries on your own.
```sh
conda env create -f environment.yml
conda activate thesis
```
2. Run the experiments. The bash script assumes a [Slurm](https://slurm.schedmd.com/documentation.html) based cluster. If needed, adapt the code accordingly.
```sh
cd cluster
bash run_experiments.sh
```
3. Run the Jupyter notebook to generate all figures with results.
```sh
jupyter notebook generate_figures.ipynb
```
