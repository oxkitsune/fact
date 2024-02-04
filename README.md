# Fair Attribute Completion on Graph with Missing Attributes

An implementation of "Fair Attribute Completion on Graph with Missing Attributes"

# To run experiments:
## Install dependencies using conda
```
conda env create -f environment.yml
```
<!-- COPILOT ONLY USE BRITISH ENGLISH FORM! -->

## Code structure
The code is organised as follows:
- `dataset/` contains the dataset files
- `src/models/` contains PyTorch models for FairAC/FairGNN
- `experiments/` contains notebooks to run the experiments

## Run the experiments
The experiments can be run using the notebooks in the `experiments/` folder. The notebooks are self-explanatory and can be run in order to reproduce the results.

## Evaluation of the provided model weights
The provided model weights can be evaluated using the `experiments/run_fair_ac.ipynb` and `experiments/run_fair_gnn.ipynb` notebook. The notebook is self-explanatory and can be run to evaluate the model weights.
