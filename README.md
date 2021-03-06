# preposition_sense_disambiguation

## Preparation

### Python Requirements

`pip install -r requirements.txt`

### Directory structure

```
170050011_170050035_170020016_assignment1
├── README.md
├── data.py
├── knn_output
├── main.py
├── mlp_output
├── report.pdf
├── requirements.txt
├── run_knn.sh
├── run_mlp.sh
├── run_svm.sh
├── setup.sh
├── svm_output
├── test_out
└── validate_format.py

4 directories, 10 files
```

### Downloading GloVe

`bash setup.sh`

### Running Model 'x' for Preposition 'y'

`python3 main.py --preposition y --model x`

where `x` can be `[svm, knn, mlp]`

### Running Model 'x' with saved hyperparameters for all the prepositions

```
bash run_mlp.sh
bash run_knn.sh
bash run_svm.sh
```
