# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: config.py 
@time: 2020/11/19 10:56
@contact: xueyao_98@foxmail.com

# config and argument
"""

from argparse import ArgumentParser, ArgumentTypeError
import os

################## Mode Dictionaries ##################

important_sentences_metrics = dict()
important_sentences_metrics[-1] = "Doc-level"
important_sentences_metrics[0] = 'bert_cls_cosine'
important_sentences_metrics[1] = 'bert_embeddings_cosine'
important_sentences_metrics[2] = 'rouge-2 f1-score'
important_sentences_metrics[3] = 'rouge-2 recall'
important_sentences_metrics[4] = 'rouge_bert'
important_sentences_metrics[5] = 'rouge-2 recall'
important_sentences_metrics[6] = 'memory based'

memory_init_strategy = dict()
memory_init_strategy[-1] = 'Do not use Memory'
memory_init_strategy[0] = 'randomly init'
memory_init_strategy[1] = 'K-means based'
memory_init_strategy[2] = 'Resume'

memory_vec_strategy = dict()
memory_vec_strategy[0] = 'BTE'
memory_vec_strategy[1] = 'RBTE'

memory_updated_strategy = dict()
memory_updated_strategy[0] = 'epoch-wise'
memory_updated_strategy[1] = 'iteration-wise'
memory_updated_strategy[2] = 'non-updated'

aggregate_strategy = dict()
aggregate_strategy[0] = 'avg pooling'
aggregate_strategy[1] = 'weighted pooling'
aggregate_strategy[2] = '[Q, S], avg pooling'
aggregate_strategy[3] = '[Q, S], weighted pooling'
aggregate_strategy[4] = '[Q, S, V], weighted pooling'
aggregate_strategy[5] = '[Q, S, V, S_Q, S_V], weighted pooling'
aggregate_strategy[6] = '[Q, S, V], avg pooling'

mode_dicts = {'sentence': important_sentences_metrics,
              'init': memory_init_strategy,
              'vec': memory_vec_strategy,
              'update': memory_updated_strategy,
              'aggregate': aggregate_strategy}

################## Input Files ##################
project_path = '/data/zhangxueyao/FactChecking/FactCheckingSearch'
if not os.path.exists(project_path):
    # Run on 233
    project_path = project_path.replace('/data/', '/home/')

query_token_files = {'weibo_old': '{}/Bert/doc_level/data/fn_token_ids.pkl'.format(project_path),
                     'weibo': '{}/dataset/weibo_checked/data/bert/fn_token_ids.pkl'.format(project_path),
                     'snopes': '{}/dataset/English/_preprocess/data/bert/fn_token_ids.pkl'.format(project_path)}
document_token_files = {'weibo_old': '{}/Bert/doc_level/data/dn_token_ids.pkl'.format(project_path),
                        'snopes': ''}
sentences_token_files = {'weibo_old': '{}/Bert/sent_level/data/dn_token_ids_sent.pkl'.format(project_path),
                         'weibo': '{}/dataset/weibo_checked/data/bert/dn_token_ids_sent.pkl'.format(project_path),
                         'snopes': '{}/dataset/English/_preprocess/data/bert/dn_token_ids_sent.pkl'.format(
                             project_path)}

BTE_memory_init_files = {
    'weibo_old': '{}/Bert/sent_level/memory/data/soft_BTE/kmeans_cluster_centers.npy'.format(project_path),
    'weibo': '{}/dataset/weibo_checked/data/soft_BTE/kmeans_cluster_centers.npy'.format(project_path),
    'snopes': '{}/dataset/English/_preprocess/data/soft_BTE/kmeans_cluster_centers.npy'.format(
        project_path)}
BTE_fn_files = {'weibo_old': '{}/Bert/sent_level/memory/data/soft_BTE/BTE_fn.pkl'.format(project_path),
                'weibo': '{}/dataset/weibo_checked/data/soft_BTE/BTE_fn.pkl'.format(project_path),
                'snopes': '{}/dataset/English/_preprocess/data/soft_BTE/BTE_fn.pkl'.format(project_path)}
BTE_dn_sent_files = {'weibo_old': '{}/Bert/sent_level/memory/data/soft_BTE/BTE_dn_sent.pkl'.format(project_path),
                     'weibo': '{}/dataset/weibo_checked/data/soft_BTE/BTE_dn_sent.pkl'.format(project_path),
                     'snopes': '{}/dataset/English/_preprocess/data/soft_BTE/BTE_dn_sent.pkl'.format(project_path)}
BTE_relevance_distance_dict_files = {
    'weibo_old': '{}/Bert/sent_level/memory/data/soft_BTE/relevance_dict.pkl'.format(project_path),
    'weibo': '{}/dataset/weibo_checked/data/soft_BTE/relevance_dict.pkl'.format(project_path),
    'snopes': '{}/dataset/English/_preprocess/data/soft_BTE/relevance_dict.pkl'.format(project_path)}
BTE_init_indicating_files = {
    'weibo_old': '{}/Bert/sent_level/memory/data/soft_BTE/memory_indicating_init.pkl'.format(project_path),
    'weibo': '{}/dataset/weibo_checked/data/soft_BTE/memory_indicating_init.pkl'.format(project_path),
    'snopes': '{}/dataset/English/_preprocess/data/soft_BTE/memory_indicating_init.pkl'.format(project_path)}

for k in ['weibo_setting1', 'weibo_setting2']:
    query_token_files[k] = query_token_files['weibo']
    sentences_token_files[k] = sentences_token_files['weibo']
    BTE_memory_init_files[k] = BTE_memory_init_files['weibo']
    BTE_fn_files[k] = BTE_fn_files['weibo']
    BTE_dn_sent_files[k] = BTE_dn_sent_files['weibo']
    BTE_relevance_distance_dict_files[k] = BTE_relevance_distance_dict_files['weibo']
    BTE_init_indicating_files[k] = BTE_init_indicating_files['weibo']


################## Parser ##################

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser(description='Bert for PDM Detection')

################## dataset ##################
parser.add_argument('--dataset', type=str, default='weibo',
                    help='[weibo, snopes]')
parser.add_argument('--weibo_setting', type=int, help='[1, 2]')

################## BM25 ##################
parser.add_argument('--topk', type=int, default=50,
                    help='input topk docs from BM25 ranking list')

################## Architecture ##################
# bert pretrained model
parser.add_argument('--pretrained_model', type=str, default='bert-base-chinese',
                    help='pretrained model name in huggingface\'s transformers')

# rouge-bert finetune
parser.add_argument('--rouge_bert_encoder_layers', type=int, default=1)
parser.add_argument('--rouge_bert_regularize', type=float, default=0.01)
parser.add_argument('--rouge_bert_model_file', type=str, default='')
parser.add_argument('--RBTE_saved_dir', type=str, default='soft_RBTE')

# architecture of base bert
parser.add_argument('--query_max_length', type=int, default=180,
                    help='max. input sequence length of query')
parser.add_argument('--bert_max_length', type=int, default=256,
                    help='max. input sequence length of model input')
parser.add_argument('--emsize', type=int, default=768,
                    help='size of word embeddings')
parser.add_argument('--bert_mlp_layers', type=int, default=5)
parser.add_argument('--finetune_front_layers', type=str2bool, default=False)
parser.add_argument('--finetune_inter_layers', type=str2bool, default=True)

# sentences selector
parser.add_argument('--selected_sentences', type=int, default=5,
                    help='selected important sentences')
parser.add_argument('--selected_sentences_mode', type=int, default=6,
                    help='selected important sentences')
parser.add_argument('--memory_init_mode', type=int, default=1,
                    help='[-1] Do not use Memory; [0] randomly init; [1] K-means based; [2] Resume')
parser.add_argument('--memory_vectorization_mode', type=int, default=0,
                    help='[0] Bert Token Embedding (BTE); [1] RougeBert Token Embedding (RBTE)')
parser.add_argument('--memory_updated_mode', type=int, default=0,
                    help='[0] epoch-wise; [1] iteration-wise; [2] Non-updated')
parser.add_argument('--memory_score_relevance_weight', type=float, default=1.0)
parser.add_argument('--memory_score_indicating_weight', type=float, default=1.0)
parser.add_argument('--memory_updated_step', type=float, default=0.3)

# sentences aggregator
parser.add_argument('--sentences_aggregate_mode', type=int, default=0,
                    help='[0] avg pooling'
                         '[1] weighted pooling'
                         '[2] [Q, S], avg pooling'
                         '[3] [Q, S], weighted pooling'
                         '[4] [Q, S, V], weighted pooling'
                         '[5] [Q, S, V, S_Q, S_V], weighted pooling'
                         '[6] [Q, S, V], avg pooling')

# networks training hyperparameters
parser.add_argument('--lr', type=float, default=5e-5,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='No. of the epoch to start training')
parser.add_argument('--resume', type=str, default='',
                    help='path to load trained model')
parser.add_argument('--resume_of_memory', type=str, default='',
                    help='path to load memory matrix')
parser.add_argument('--evaluate', type=str2bool, default=False,
                    help='only use for evaluating')
parser.add_argument('--save', type=str, default='./ckpts/',
                    help='folder to save the final model')

# random seed
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

################## Resources ##################
parser.add_argument('--device', default='cpu')
parser.add_argument('--fp16', type=str2bool, default=True,
                    help='use fp16 for training')
parser.add_argument('--local_rank', type=int, default=-1)

################## Debug ##################
parser.add_argument('--debug', type=str2bool, default=False)
