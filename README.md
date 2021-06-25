# Pipeline

## Weibo Dataset

### BM25

Calculate the BM25 ranking matrix, `MTM/preprocess/BM25/data/bm25_scores_(#Claims, #Articles).npy`, by `MTM/preprocess/BM25/Weibo.ipynb`.

### Datasets Splits

`MTM/dataset/Weibo/splits/data_splits.ipynb`

1. Split by event
2. Split by claim

### Tokenize

`MTM/preprocess/tokenize/run.sh`

### ROT

`MTM/preprocess/ROT/run.sh`

1. Prepare RougeBert's Training data
2. Training

## Twitter Dataset