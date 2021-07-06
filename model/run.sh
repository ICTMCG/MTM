CUDA_VISIBLE_DEVICES=1 python main.py --debug False --save 'ckpts/Weibo' \
--dataset 'Weibo' --pretrained_model 'bert-base-chinese' \
--rouge_bert_model_file '../preprocess/ROT/ckpts/Weibo/9.pt' \
--memory_init_file '../preprocess/PMB/data/Weibo/kmeans_cluster_centers.npy' \
--memory_updated_step 0.3 --lambdaQ 0.6 --lambdaP 0.4 \
--selected_sentences 3 \
--lr 5e-5 --epochs 10 --batch_size 32 \
