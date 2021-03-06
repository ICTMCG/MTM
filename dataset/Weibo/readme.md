# Weibo Dataset

## Notice

Please note that **the dataset may not be used for any purpose other than research**.

## Access

You will be shared the dataset by email after an ["Application to Use the Chinese Dataset for Detecting Previously Fact-Checked Claim"](https://forms.office.com/r/FWVetbSM7p) has been submitted.

## Usage

### The Raw Dataset

The raw dataset can be seen in `raw` folder, including a claims' file and a articles' file:

#### Claim

- Saved in `raw/FN_11934_filtered.json`.
  - the `_id` identifies the unique id of the claim.
  - the `event` identifies the unique id of the event that the claim belongs to.
  - the `time` and `time_format` represent the timestamp and the datetime of the claim.
  - the `content` is the content of a claim.
  - the `content_ocr` (if owns) is the OCR's result of a claim's images.
  - the `content_all` is combined by `content` and `content_ocr`.
  - the `debunking_ids` identifies the corresponding articles' `_id` in `DN_27505_filtered.json`.
- The num of claims is 11934.

#### Articles

- Saved in `raw/DN_27505_filtered.json`.
  - the `_id` identifies the unique id of the article.
  - the `time` and `time_format` represent the timestamp and the datetime of the claim.
  - the `content` is the content of an article.
  - the `content_ocr` (if owns) is the OCR's result of an article's images.
  - the `content_all` is combined by `content` and `content_ocr`, which is organized as sentences' list. 
- The num of articles is 27505.

### Dataset Splits

In `splits` folder, we splitted the datasets on the basis of the ranking results of Stage 1 (BM25). The results are saved in `splits/data`. These `top50`-prefixed files are CSV-friendly format, and there are five columns split by `\t` of every file:

1. `qid`: the unique id of the query claim.
2. `qidx`: the index of the query claim.
3. `did`: the unique id(s) of the debunking article(s)
4. `didx`: the index(es) of the debunking article(s)
5. `label`: the label(s) of the "query, article(s)" pair(s), where 1 for "relevant" and 0 for "irrelevant".

You can refer to `MTM/model/DatasetLoader.py` to load them.



If you want to reproduce the dataset splits (note that is not necessary):

1. To calculate the BM25 ranking matrix,

   ```
   cd MTM/preprocess/BM25
   mkdir data
   ```

   then you need to run `Weibo.ipynb` to get the `data/bm25_scores_(11934, 27505).npy`.

2. To split the datasets,

   ```
   cd MTM/dataset/Weibo/splits
   ```

   then you need to run `data_splits.ipynb` to get the `top50`-prefixed files in `data`.

