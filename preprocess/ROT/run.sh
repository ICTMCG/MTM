# Prepare Training Data
python prepare_for_rouge.py --dataset Weibo --pretrained_model bert-base-chinese

# Pretrain ROUGE-guided Transformer (ROT)
CUDA_VISIBLE_DEVICES=1 python main.py --debug False \
--dataset Weibo --pretrained_model bert-base-chinese --save './ckpts/Weibo' \
--rouge_bert_encoder_layers 1 --rouge_bert_regularize 0.01 \
--fp16 True

# Get Embeddings
CUDA_VISIBLE_DEVICES=3 python get_embeddings.py \
--rouge_bert_model_file './ckpts/Weibo/9.pt' \
--batch_size 1024 --embeddings_type static
