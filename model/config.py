# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: config.py 
@time: 2020/11/19 10:56
@contact: xueyao_98@foxmail.com

# config and argument
"""

from argparse import ArgumentParser, ArgumentTypeError

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
parser.add_argument('--dataset', type=str, default='Weibo',
                    help='[Weibo, Twitter]')

################## Architecture ##################
# bert pretrained model
parser.add_argument('--pretrained_model', type=str, default='bert-base-chinese',
                    help='pretrained model name in huggingface\'s transformers')

# rouge-bert
parser.add_argument('--rouge_bert_model_file', type=str, help='pretrained ROT')
parser.add_argument('--rouge_bert_encoder_layers', type=int, default=1)
parser.add_argument('--rouge_bert_regularize', type=float, default=0.01)

# memory related
parser.add_argument('--memory_init_file', type=str, help='PMB')
parser.add_argument('--claim_sentence_distance_file', type=str,
                    default='./data/claim_sentence_distance.pkl')
parser.add_argument('--pattern_sentence_distance_init_file',
                    type=str, default='./data/pattern_sentence_distance_init.pkl',
                    help='The scores will be updated when PMB is updated.')
parser.add_argument('--memory_updated_step', type=float, default=0.3)

# key sentences selection
parser.add_argument('--selected_sentences', type=int, default=3)
parser.add_argument('--lambdaQ', type=float, default=0.6,
                    help='weight of claim-sentence')
parser.add_argument('--lambdaP', type=float, default=0.4,
                    help='weight of pattern-sentence')

# bert
parser.add_argument('--bert_max_length', type=int, default=256)
parser.add_argument('--query_max_length', type=int, default=180)
parser.add_argument('--emsize', type=int, default=768)
parser.add_argument('--finetune_front_layers', type=str2bool, default=False)
parser.add_argument('--finetune_inter_layers', type=str2bool, default=True)
parser.add_argument('--bert_mlp_layers', type=int, default=5)

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
parser.add_argument('--save', type=str, default='./ckpts/debug',
                    help='folder to save the final model')

# random seed
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

################## Resources ##################
parser.add_argument('--device', default='cpu')
parser.add_argument('--fp16', type=str2bool, default=True,
                    help='use fp16 for training')

################## Debug ##################
parser.add_argument('--debug', type=str2bool, default=False)
