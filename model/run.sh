# # === Weibo ===
# CUDA_VISIBLE_DEVICES=3 python main.py --debug False --save 'ckpts/Weibo' \
# --dataset 'Weibo' --pretrained_model 'bert-base-chinese' \
# --rouge_bert_model_file '../preprocess/ROT/ckpts/Weibo/9.pt' \
# --memory_init_file '../preprocess/PMB/data/Weibo/kmeans_cluster_centers.npy' \
# --claim_sentence_distance_file './data/Weibo/claim_sentence_distance.pkl' \
# --pattern_sentence_distance_init_file './data/Weibo/pattern_sentence_distance_init.pkl' \
# --memory_updated_step 0.3 --lambdaQ 0.6 --lambdaP 0.4 \
# --selected_sentences 3 \
# --lr 5e-6 --epochs 10 --batch_size 32 \

# === Twitter ===
CUDA_VISIBLE_DEVICES=0 python main.py --debug False --save 'ckpts/Twitter' \
--dataset 'Twitter' --pretrained_model 'bert-base-uncased' \
--rouge_bert_model_file '../preprocess/ROT/ckpts/Twitter/8.pt' \
--memory_init_file '../preprocess/PMB/data/Twitter/kmeans_cluster_centers.npy' \
--claim_sentence_distance_file './data/Twitter/claim_sentence_distance.pkl' \
--pattern_sentence_distance_init_file './data/Twitter/pattern_sentence_distance_init.pkl' \
--memory_updated_step 0.3 --lambdaQ 0.6 --lambdaP 0.4 \
--selected_sentences 5 \
--lr 1e-4 --epochs 10 --batch_size 16 \
