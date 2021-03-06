# preposition_sense_disambiguation

## Preparation

### Python Requirements

`pip install -r requirements.txt`

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
