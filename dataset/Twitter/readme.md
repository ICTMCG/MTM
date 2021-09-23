# Twitter Dataset

## The Original Dataset

The original twitter dataset is available at https://github.com/nguyenvo09/EMNLP2020.

```
@inproceedings{vo2020facts,
	title={Where Are the Facts? Searching for Fact-checked Information to Alleviate the Spread of Fake News},
	author={Vo, Nguyen and Lee, Kyumin},
	booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)},
	year={2020}
}
```

## Preprocess

To unify the Twitter dataset format with the Weibo dataset, we re-organize the original dataset. And the preprocessed dataset is saved in `raw` and `splits` dirs.

### The Raw Dataset

The raw dataset can be seen in `raw` folder, including a claims' file and a articles' file:

#### Claim

- Saved in `raw/FN_10003.json`.
  - the `_id` identifies the unique id of the claim.
  - the `content` is the content of a claim.
  - the `debunking_ids` identifies the corresponding articles' `_id` in `DN_1703.json`.
- The num of claims is 10003.

#### Articles

- Saved in `raw/DN_1703.json`.
  - the `_id` identifies the unique id of the article.
  - the `url` represents the original URL of the article.
  - the `content` is the content of an article, which is organized as sentences' list. 
- The num of articles is 1703.

### Dataset Splits

We follow the original dataset splits, and just re-organize the data format. The results are saved in `splits/data`. These `top50`-prefixed files are CSV-friendly format, which you can refer to `MTM/model/DatasetLoader.py` to load them.