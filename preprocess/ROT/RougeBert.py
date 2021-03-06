# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: RougeBert.py 
@time: 2020/11/17 14:53
@contact: xueyao_98@foxmail.com

# RougeBert: Bert's Embeddings + Bert's first layer of encoder
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import pickle
import os
import sys


class RougeBert(nn.Module):
    def __init__(self, args):
        super(RougeBert, self).__init__()

        self.args = args

        # Loading tokens
        FN_tokens_file = '../tokenize/data/{}/FN_{}.pkl'.format(
            args.dataset, args.pretrained_model)
        DN_tokens_file = '../tokenize/data/{}/DN_{}.pkl'.format(
            args.dataset, args.pretrained_model)
        self.q_tokens = pickle.load(open(FN_tokens_file, 'rb'))
        self.d_tokens_sentences = pickle.load(open(DN_tokens_file, 'rb'))

        # Loading pretrained model's some layers
        pretrained_bert = BertModel.from_pretrained(args.pretrained_model)
        vocab_size = pretrained_bert.state_dict(
        )['embeddings.word_embeddings.weight'].shape[0]
        self.model = BertModel(BertConfig(
            num_hidden_layers=args.rouge_bert_encoder_layers, vocab_size=vocab_size))

        pretrained_model_dict = dict()
        for name in self.model.state_dict():
            if 'pooler' in name:
                pretrained_model_dict[name] = self.model.state_dict()[name]
            else:
                pretrained_model_dict[name] = pretrained_bert.state_dict()[
                    name]
        self.model.load_state_dict(pretrained_model_dict)

        for name, param in self.model.named_parameters():
            param.requires_grad = True

        fixed_layers = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                fixed_layers.append(name)

        print("#Fixed layers: {}/{}：\n{}".format(
            len(fixed_layers), len(self.model.state_dict()), fixed_layers))
        print('\n*************************************************\n')

        # bert input length
        self.maxlen = args.bert_max_length
        self.query_maxlen = args.query_max_length
        self.doc_maxlen = self.maxlen - args.query_max_length - 3

        # other layers
        self.fc1 = nn.Linear(args.emsize, 2, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, qidxs, didxs, sidxs):
        queries = [self.q_tokens[q.item()] for q in qidxs]
        sentences = [self.d_tokens_sentences[didxs[i].item()][sidxs[i].item()]
                     for i in range(len(qidxs))]

        input_ids, attention_mask, token_type_ids = zip(
            *[self._encode(queries[i], sentences[i]) for i in range(len(queries))])
        input_ids, attention_mask, token_type_ids = self._tensorize(input_ids), self._tensorize(
            attention_mask), self._tensorize(token_type_ids)

        _, pooled = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        a = self.fc1(pooled)
        b = self.dropout(a)
        out = torch.sigmoid(b)

        return out

    def _encode(self, q, d):
        q = q[:self.query_maxlen]
        d = d[:self.doc_maxlen]

        padding_length = self.maxlen - (len(q) + len(d) + 3)

        if len(d) != 0:
            attention_mask = [1] * (len(q) + len(d) + 3) + [0] * padding_length
        else:
            attention_mask = [1] * (len(q) + 1) + [0] * (self.maxlen - (len(q) + 1))

        input_ids = [101] + q + [102] + d + [102] + [103] * padding_length
        token_type_ids = [0] * (len(q) + 2) + [1] * (self.maxlen - len(q) - 2)

        return input_ids, attention_mask, token_type_ids

    def _tensorize(self, l):
        return torch.as_tensor(l, dtype=torch.long, device=self.args.device)
