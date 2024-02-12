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
| Dataset   | Method      | Acc ↑           | AUC ↑            | ΔSP ↓            | ΔEO ↓            | ΔSP+ΔEO ↓        | Consistency ↑   |
|-----------|-------------|-----------------|------------------|------------------|------------------|------------------|-----------------|
| NBA       | FairAC | 66.51 ± 1.09    | 75.69 ± 1.31     | **0.09 ± 0.08**  | **0.10 ± 0.00**  | **0.19 ± 0.08**  | **2.64 ± 0.00** |
| Pokec-z   | FairAC | 65.33 ± 0.30    | 71.20 ± 1.74     | **0.55 ± 0.10**  | **0.13 ± 0.15**  | **0.68 ± 0.09**  | 41.33 ± 0.00    |
| Pokec-n   | FairAC | 67.00 ± 1.93    | 72.57 ± 1.68     | **0.11 ± 0.06**  | 0.47 ± 0.81      | **0.58 ± 0.76**  | **45.94 ± 0.02** |


These result can be reproduced by running the `*_3_seeds.ipynb` notebooks under `experiments/fair_ac`
