# === Weibo === 
# Prepare Training Data
python prepare_for_rouge.py --dataset Weibo --pretrained_model bert-base-chinese

# Pretrain ROUGE-guided Transformer (ROT)
CUDA_VISIBLE_DEVICES=1 python main.py --debug False \
--dataset Weibo --pretrained_model bert-base-chinese --save './ckpts/Weibo' \
--rouge_bert_encoder_layers 1 --rouge_bert_regularize 0.01 \
--fp16 True

# Get Embeddings
CUDA_VISIBLE_DEVICES=3 python get_embeddings.py \
--dataset Weibo --pretrained_model bert-base-chinese \
--rouge_bert_model_file './ckpts/Weibo/9.pt' \
--batch_size 1024 --embeddings_type static

# === Twitter === 
# Prepare Training Data
python prepare_for_rouge.py --dataset Twitter --pretrained_model bert-base-uncased

# Pretrain ROUGE-guided Transformer (ROT)
CUDA_VISIBLE_DEVICES=0 python main.py --debug False \
--dataset Twitter --pretrained_model bert-base-uncased --save './ckpts/Twitter' \
--rouge_bert_encoder_layers 1 --rouge_bert_regularize 0.05 \
--fp16 True

# Get Embeddings
CUDA_VISIBLE_DEVICES=0 python get_embeddings.py \
--dataset Twitter --pretrained_model bert-base-uncased \
--rouge_bert_model_file './ckpts/Twitter/8.pt' \
--batch_size 1024 --embeddings_type static
