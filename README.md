# Reproducibility study of FairAC
An implementation of "Fair Attribute Completion on Graph with Missing Attributes" paper.

# To run experiments:
## Install dependencies using conda
```
conda env create -f environment.yml
```

## Code structure
The code is organised as follows:
- `dataset/` contains the dataset files
- `src/models/` contains PyTorch models for FairAC/FairGNN
- `experiments/` contains notebooks to run the experiments

## Run the experiments
The experiments can be run using the notebooks in the `experiments/` folder. The notebooks are self-explanatory and can be run in order to reproduce the results.

## Evaluation of the provided model weights
We provide the pre-trained weights from our experiments in `experiments/logs`.
These weights can be evaluated using the `experiments/run_fair_ac.ipynb` notebook. The notebook is self-explanatory and can be run to evaluate the model weights.

## Deepwalk embeddings
Several deepwalk embeddings are provided already. If more are needed they can be created by running:
```bash
python src/create_deepwalk_emb.py
```


## Results
| Dataset | Method | M | Acc ↑ | AUC ↑ | ΔSP ↓ | ΔEO ↓ | ΔSP+ΔEO ↓ | Consistency ↑ |
|---------|--------|---|-------|-------|-------|-------|-----------|--------------|
| **NBA** | GCN | ✔ | 66.98 ± 1.18 | **76.15 ± 1.40** | 0.14 ± 0.13 | 0.57 ± 0.06 | 0.71 ± 0.18 | **2.64 ± 0.00** |
|         | ALFR | ✖ | 64.3±1.3 | 71.5±0.3 | 2.3±0.9 | 3.2±1.5 | 5.5±2.4 | - |
|         | ALFR-e | ✖ | 66.0±0.4 | 72.9±1.0 | 4.7±1.8 | 4.7±1.7 | 9.4±3.4 | - |
|         | Debias | ✖ | 63.1±1.1 | 71.3±0.7 | 2.5±1.5 | 3.1±1.9 | 5.6±3.4 | - |
|         | Debias-e | ✖ | 65.6±2.4 | 72.9±1.2 | 5.3±0.9 | 3.1±1.3 | 8.4±2.2 | - |
|         | FCGE | ✖ | 66.0±1.5 | 73.6±1.5 | 2.9±1.0 | 3.0±1.2 | 5.9±2.2 | - |
|         | FairGNN | ✔ | **68.39 ± 3.12** | 74.29 ± 1.19 | 2.81 ± 4.01 | 3.00 ± 4.07 | 5.81 ± 8.08 | **2.64 ± 0.00** |
|         | FairAC (Ours) | ✔ |66.51 ± 1.09 | 75.69 ± 1.31 | **0.09 ± 0.08** | **0.10 ± 0.00** | **0.19 ± 0.08** | **2.64 ± 0.00** |
| **Pokec-z** | GCN | ✔ | 65.10 ± 0.24 | 68.42 ± 0.12 | 1.72 ± 1.17 | 1.37 ± 0.51 | 3.08 ± 1.68 | **41.35 ± 0.01** |
|             | ALFR | ✖ | 65.4±0.4 | 71.3±0.3 | 2.8±0.5 | 1.1±0.4 | 3.9±0.9 | - |
|             | ALFR-e | ✖ | 68.0±0.6 | 74.0±0.7 | 5.8±0.4 | 2.8±0.8 | 8.6±1.2 | - |
|             | Debias | ✖ | 65.2±0.7 | 71.4±0.6 | 1.9±0.6 | 1.9±0.4 | 3.8±1.0 | - |
|             | Debias-e | ✖ | 67.5±0.7 | 74.2±0.7 | 4.7±1.0 | 3.0±1.4 | 7.7±2.4 | - |
|             | FCGE | ✖ | 65.9±0.2 | 71.0±0.2 | 3.1±0.5 | 1.7±0.6 | 4.8±1.1 | - |
|             | FairGNN | ✔ | **68.16 ± 0.59** | **75.67 ± 0.52** | 1.56 ± 0.45 | 3.17 ± 1.07 | 4.73 ± 1.47 | **41.35 ± 0.01** |
|             | FairAC (Ours) | ✔ | 65.33 ± 0.30 | 71.20 ± 1.74 | **0.55 ± 0.10** | **0.13 ± 0.15** | **0.68 ± 0.09** | 41.33 ± 0.00 |
| **Pokec-n** | GCN | ✔ | **67.88 ± 1.46** | **72.86 ± 1.44** | 3.22 ± 1.29 | 5.93 ± 2.76 | 9.15 ± 4.05 | 45.93 ± 0.00 |
|             | ALFR | ✖ | 63.1±0.6 | 67.7±0.5 | 3.05±0.5 | 3.9±0.6 | 3.95±1.1 | - |
|             | ALFR-e | ✖ | 66.2±0.4 | 71.9±1.0 | 4.1±1.8 | 4.6±1.7 | 8.7±3.5 | - |
|             | Debias | ✖ | 62.6±1.1 | 67.9±0.7 | 2.4±1.5 | 2.6±1.9 | 5.0±3.4 | - |
|             | Debias-e | ✖ | 65.6±2.4 | 71.7±1.2 | 3.6±0.9 | 4.4±1.3 | 8.0±2.2 | - |
|             | FCGE | ✖ | 64.8±1.5 | 69.5±1.5 | 4.1±1.0 | 5.5±1.2 | 9.6±2.2 | - |
|             | FairGNN | ✔ | 67.06 ± 0.37 | 71.58 ± 2.58 | 0.55 ± 0.50 | **0.30 ± 0.20** | 0.85 ± 0.31 | 45.93 ± 0.00 |
|             | FairAC (Ours) | ✔ | 67.00 ± 1.93 | 72.57 ± 1.68 | **0.11 ± 0.06** | 0.47 ± 0.81 | **0.58 ± 0.76** | **45.94 ± 0.02** |

> Table 1: Comparison of FairAC with FairGNN on the nba, pokec-z and pokec-n dataset. 
> - The methods are applied on the GCN classifier, and the values for the baselines are taken from the original paper. 
> - The values for FairAC, FairGNN and GCN are taken from our experiments, that can be recreated using the notebooks under `experiments/`. 
> - The values consist of the mean and standard deviation of the metric over 3 runs on the seeds 40, 41 and 42. 
> - The best results are denoted in bold.
