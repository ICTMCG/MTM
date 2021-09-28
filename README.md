# MTM

This is the official repository of the paper:

> **Article Reranking by Memory-enhanced Key Sentence Matching for Detecting Previously Fact-checked Claims.**
>
> Qiang Sheng, Juan Cao, Xueyao Zhang, Xirong Li, and Lei Zhong.
>
> *Proceedings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)*
>
> [PDF](https://aclanthology.org/2021.acl-long.425.pdf) / [Poster](https://sheng-qiang.github.io/data/MTM-Poster.pdf) / [Code](https://github.com/ICTMCG/MTM) / [Chinese Dataset](https://forms.office.com/r/FWVetbSM7p) / [Chinese Blog 1](https://zhuanlan.zhihu.com/p/393615707) / [Chinese Blog 2](https://mp.weixin.qq.com/s/YIfYlPEIXrF3dLvFHfqENQ)

## Datasets

There are two experimental datasets, including the [Twitter Dataset](https://github.com/ICTMCG/MTM/tree/main/dataset/Twitter), and the firstly proposed [Weibo Dataset](https://github.com/ICTMCG/MTM/tree/main/dataset/Weibo). Note that you can download the Weibo Dataset only after an  ["Application to Use the Chinese Dataset for Detecting Previously Fact-Checked Claim"](https://forms.office.com/r/FWVetbSM7p) has been submitted.

## Code

### Key Requirements

```
python==3.6.10
torch==1.6.0
torchvision==0.7.0
transformers==3.2.0
```

## Usage for Weibo Dataset

After you download the dataset (the way to access is described [here](https://github.com/ICTMCG/MTM/tree/main/dataset/Weibo)), move the `FN_11934_filtered.json` and `DN_27505_filtered.json` into the path `MTM/dataset/Weibo/raw`:

```
mkdir MTM/dataset/Weibo/raw
mv FN_11934_filtered.json MTM/dataset/Weibo/raw
mv DN_27505_filtered.json MTM/dataset/Weibo/raw
```

### Preparation

#### Tokenize 

```
cd MTM/preprocess/tokenize
sh run_weibo.sh
```

#### ROT

```
cd MTM/preprocess/ROT
```

You can refer to the `run_weibo.sh`, which includes three steps:

1. Prepare RougeBert's Training data:

   ```
   python prepare_for_rouge.py --dataset Weibo --pretrained_model bert-base-chinese
   ```

2. Training:

   ```
   CUDA_VISIBLE_DEVICES=0 python main.py --debug False \
   --dataset Weibo --pretrained_model bert-base-chinese --save './ckpts/Weibo' \
   --rouge_bert_encoder_layers 1 --rouge_bert_regularize 0.01 \
   --fp16 True
   ```

   then you can get `ckpts/Weibo/[EPOCH].pt`.

3. Vectorize the claims and articles (get embeddings):

   ```
   CUDA_VISIBLE_DEVICES=0 python get_embeddings.py \
   --dataset Weibo --pretrained_model bert-base-chinese \
   --rouge_bert_model_file './ckpts/Weibo/[EPOCH].pt' \
   --batch_size 1024 --embeddings_type static
   ```

#### PMB

```
cd MTM/preprocess/PMB
```

1. Prepare the clustering data:

   ```
   mkdir data
   mkdir data/Weibo
   ```

   and you can get `data/Weibo/clustering_training_data_[TS_SMALL]<dist<[TS_LARGE].pkl` after running `calculate_init_thresholds.ipynb`.

2. Kmeans clustering. You can refer to the `run_weibo.sh`:

   ```
   python kmeans_clustering.py --dataset Weibo --pretrained_model bert-base-chinese --clustering_data_file 'data/Weibo/clustering_training_data_[TS_SMALL]<dist<[TS_LARGE].pkl'
   ```

   then you can get `data/Weibo/kmeans_cluster_centers.npy`.

Besides, it is available to see some cases of key sentences selection in `key_sentences_selection_cases_Weibo.ipynb`.

### Training and Inferring

```
cd MTM/model
mkdir data
mkdir data/Weibo
```

You can refer to the `run_weibo.sh`:

```
CUDA_VISIBLE_DEVICES=0 python main.py --debug False --save 'ckpts/Weibo' \
--dataset 'Weibo' --pretrained_model 'bert-base-chinese' \
--rouge_bert_model_file '../preprocess/ROT/ckpts/Weibo/[EPOCH].pt' \
--memory_init_file '../preprocess/PMB/data/Weibo/kmeans_cluster_centers.npy' \
--claim_sentence_distance_file './data/Weibo/claim_sentence_distance.pkl' \
--pattern_sentence_distance_init_file './data/Weibo/pattern_sentence_distance_init.pkl' \
--memory_updated_step 0.3 --lambdaQ 0.6 --lambdaP 0.4 \
--selected_sentences 3 \
--lr 5e-6 --epochs 10 --batch_size 32 \
```

then the results and ranking reports will be saved in `ckpts/Weibo`.

## Usage for Twitter Dataset

The description of the dataset can be seen at [here](https://github.com/ICTMCG/MTM/tree/main/dataset/Twitter).

### Preparation

#### Tokenize 

```
cd MTM/preprocess/tokenize
sh run_twitter.sh
```

#### ROT

```
cd MTM/preprocess/ROT
```

You can refer to the `run_twitter.sh`, which includes three steps:

1. Prepare RougeBert's Training data:

   ```
   python prepare_for_rouge.py --dataset Twitter --pretrained_model bert-base-uncased
   ```

2. Training:

   ```
   CUDA_VISIBLE_DEVICES=0 python main.py --debug False \
   --dataset Twitter --pretrained_model bert-base-uncased --save './ckpts/Twitter' \
   --rouge_bert_encoder_layers 1 --rouge_bert_regularize 0.05 \
   --fp16 True
   ```

   then you can get `ckpts/Twitter/[EPOCH].pt`.

3. Vectorize the claims and articles (get embeddings):

   ```
   CUDA_VISIBLE_DEVICES=0 python get_embeddings.py \
   --dataset Twitter --pretrained_model bert-base-uncased \
   --rouge_bert_model_file './ckpts/Twitter/[EPOCH].pt' \
   --batch_size 1024 --embeddings_type static
   ```

#### PMB

```
cd MTM/preprocess/PMB
```

1. Prepare the clustering data:

   ```
   mkdir data
   mkdir data/Twitter
   ```

   and you can get `data/Twitter/clustering_training_data_[TS_SMALL]<dist<[TS_LARGE].pkl` after running `calculate_init_thresholds.ipynb`.

2. Kmeans clustering. You can refer to the `run_twitter.sh`:

   ```
   python kmeans_clustering.py --dataset Twitter --pretrained_model bert-base-uncased --clustering_data_file 'data/Twitter/clustering_training_data_[TS_SMALL]<dist<[TS_LARGE].pkl'
   ```

   then you can get `data/Twitter/kmeans_cluster_centers.npy`.

Besides, it is available to see some cases of key sentences selection in `key_sentences_selection_cases_Twitter.ipynb`.

### Training and Inferring

```
cd MTM/model
mkdir data
mkdir data/Twitter
```

You can refer to the `run_twitter.sh`:

```
CUDA_VISIBLE_DEVICES=0 python main.py --debug False --save 'ckpts/Twitter' \
--dataset 'Twitter' --pretrained_model 'bert-base-uncased' \
--rouge_bert_model_file '../preprocess/ROT/ckpts/Twitter/[EPOCH].pt' \
--memory_init_file '../preprocess/PMB/data/Twitter/kmeans_cluster_centers.npy' \
--claim_sentence_distance_file './data/Twitter/claim_sentence_distance.pkl' \
--pattern_sentence_distance_init_file './data/Twitter/pattern_sentence_distance_init.pkl' \
--memory_updated_step 0.3 --lambdaQ 0.6 --lambdaP 0.4 \
--selected_sentences 5 \
--lr 1e-4 --epochs 10 --batch_size 16 \
```

then the results and ranking reports will be saved in `ckpts/Twitter`.

## Citation

```
@inproceedings{MTM,
  author    = {Qiang Sheng and
               Juan Cao and
               Xueyao Zhang and
               Xirong Li and
               Lei Zhong},
  title     = {Article Reranking by Memory-Enhanced Key Sentence Matching for Detecting
               Previously Fact-Checked Claims},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational
               Linguistics and the 11th International Joint Conference on Natural
               Language Processing, {ACL/IJCNLP} 2021},
  pages     = {5468--5481},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.acl-long.425},
  doi       = {10.18653/v1/2021.acl-long.425},
}
```

