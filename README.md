**[Notes]** The repo may be incomplete and some of the code is a bit messy. We will improve in the future. Readme will also include more details. Coming soon stay tuned :)

# Pipeline

## Weibo Dataset

### Preprocess

#### BM25

Calculate the BM25 ranking matrix, `MTM/preprocess/BM25/data/bm25_scores_(#Claims, #Articles).npy`, by `MTM/preprocess/BM25/Weibo.ipynb`.

#### Datasets Splits

`MTM/dataset/Weibo/splits/data_splits.ipynb`

1. Split by event
2. Split by claim

#### Tokenize

`sh MTM/preprocess/tokenize/run.sh`

#### ROT

`sh MTM/preprocess/ROT/run.sh`

1. Prepare RougeBert's Training data
2. Training, get `MTM/preprocess/ROT/ckpts/Weibo/[EPOCH].pt`
3. Vectorize the claims and articles (get embeddings)

#### PMB

1. Prepare the clustering data, `MTM/preprocess/PMB/data/Weibo/clustering_training_data_[TS_SMALL]<dist<[TS_LARGE].pkl`, by `MTM/preprocess/PMB/calculate_init_thresholds.ipynb`.
2. Kmeans clustering: `MTM/preprocess/PMB/run.sh`, get `MTM/preprocess/PMB/data/Weibo/kmeans_cluster_centers.npy`
3. See some cases of key sentences selection: `MTM/preprocess/PMB/key_sentences_selection_cases.ipynb`

### Training

`sh MTM/model/run.sh`

## Twitter Dataset