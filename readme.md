This repository hosts data and code used in the paper 'Aggregating Pairwise Semantic Differences for Few-Shot Claim Verification'.

## Environment setup
Installing a dedicated Python environment is recommended with the following code:

`conda env create -n seed --file environment.yml`

## Data
[fever](data/fever) contains the data used for three-way claim verification on FEVER. It is the original test set of the FEVER datset.

[fever2](data/fever2) contains the data used for binary claim verification on FEVER. It is kindly offered by the authors of 'Towards Few-Shot Fact-Checking via Perplexity'.

[scifact](data/scifact) contains the data used for three-way calim verification on SCIFACT. It is the original files of the SCIFACT dataset.

## SEED
We apply SEED to binary FEVER, three-way FEVER, three-way SCIFACT with [seed_fever_binary](seed_fever_binary.py), [seed_fever](seed_fever.py) and [seed_scifact](seed_scifact.py) respectively.

To reproduce the experiments reported in the paper, use [run_seed.sh](run_seed.sh) with specified task name at the beginning. 'bfever' stands for binary FEVER; 'fever' stands for three-way FEVER; 'scifact' stands for three-way SCIFACT.

## finetuning 
To reproduce the finetuning experiments reported in the paper, use [run_finetuning.sh](run_finetuning.sh) with specified task name at the beginning. 'bfever' stands for binary FEVER; 'fever' stands for three-way FEVER; 'scifact' stands for three-way SCIFACT.


