# === Twitter ===
CUDA_VISIBLE_DEVICES=1 python main.py --debug False --save 'ckpts/Twitter_5e-5' \
--dataset 'Twitter' --pretrained_model 'bert-base-uncased' \
--rouge_bert_model_file '../preprocess/ROT/ckpts/Twitter/8.pt' \
--memory_init_file '../preprocess/PMB/data/Twitter/kmeans_cluster_centers.npy' \
--claim_sentence_distance_file './data/Twitter/claim_sentence_distance.pkl' \
--pattern_sentence_distance_init_file './data/Twitter/pattern_sentence_distance_init.pkl' \
--memory_updated_step 0.3 --lambdaQ 0.6 --lambdaP 0.4 \
--selected_sentences 5 \
--lr 5e-5 --epochs 10 --batch_size 16 \
