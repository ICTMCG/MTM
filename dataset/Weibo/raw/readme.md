# Weibo Dataset

## Claim

- Saved in `FN_11934_filtered.json`.
  - the `content` value is the content of a claim.
  - the `content_ocr` (if owns) value is the OCR's result of a claim's images.
  - the `content_all` value is combined by `content` and `content_ocr`.
  - the `debunking_ids` value identifies the corresponding articles' `_id` in `DN_27505_filtered.json`.
- The num of claims is 11934.

## Articles

- Saved in `DN_27505_filtered.json`.
  - the `content` value is the content of an article.
  - the `content_ocr` (if owns) value is the OCR's result of an article's images.
  - the `content_all` value is combined by `content` and `content_ocr`, which is organized as sentences' list. 
- The num of articles is 27505.