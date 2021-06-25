# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@time: 2020/11/11 15:38
@contact: xueyao_98@foxmail.com

# rouge pre-process
"""

from DatasetLoader import DatasetLoader
from tqdm import tqdm
import rouge
import pickle
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
import pandas as pd
from tqdm import tqdm


def add_elem_to_dict(qidx, didx):
    fn_t = fn_tokens[qidx]
    dn_sent_t = dn_sent_tokens[didx]

    scores = rg.get_scores([' '.join(map(str, dn_t)) for dn_t in dn_sent_t],
                           [' '.join(map(str, fn_t)) for _ in dn_sent_t])

    if qidx not in rouge_rank_dict.keys():
        rouge_rank_dict[qidx] = {didx: scores}
    else:
        rouge_rank_dict[qidx][didx] = scores


if __name__ == '__main__':
    parser = ArgumentParser(description='Prepare RougeBert\'s Traning Data')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pretrained_model', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    pretrained_model = args.pretrained_model
    tokens_dir = os.path.join('../tokenize/data/{}'.format(dataset))
    save_dir = './data/{}'.format(dataset)

    rouge_labels_file = os.path.join(save_dir, 'rouge2_labels.pkl')

    if not os.path.exists(rouge_labels_file):
        rouge_scores_file = os.path.join(save_dir, 'rouge2_scores_raw.pkl')

        with open(os.path.join(tokens_dir, 'FN_{}.pkl'.format(pretrained_model)), 'rb') as f:
            fn_tokens = pickle.load(f)
        with open(os.path.join(tokens_dir, 'DN_{}.pkl'.format(pretrained_model)), 'rb') as f:
            dn_sent_tokens = pickle.load(f)

        train_dataset = DatasetLoader('train.line')
        val_dataset = DatasetLoader('val')
        test_dataset = DatasetLoader('test')

        train_loader = DataLoader(train_dataset, batch_size=1)
        val_loader = DataLoader(val_dataset, batch_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)

        rg = rouge.Rouge(metrics=['rouge-2'])

        rouge_rank_dict = dict()

        for _, qidx, _, didx, _ in tqdm(train_loader):
            add_elem_to_dict(qidx.item(), didx.item())

        for _, qidx, _, didxs, _ in tqdm(val_loader):
            for didx in didxs:
                add_elem_to_dict(qidx.item(), didx.item())

        for _, qidx, _, didxs, _ in tqdm(test_loader):
            for didx in didxs:
                add_elem_to_dict(qidx.item(), didx.item())

        with open(rouge_scores_file, 'wb') as f:
            pickle.dump(rouge_rank_dict, f)

        scores = rouge_rank_dict

        print()
        print(type(scores), len(scores))
        # print(scores[0])
        print()

        labels = dict()
        for qid, qid_dict in tqdm(scores.items()):
            labels[qid] = dict()
            for did, items in qid_dict.items():
                labels[qid][did] = [
                    (item['rouge-2']['p'], item['rouge-2']['r']) for item in items]

        with open(rouge_labels_file, 'wb') as f:
            pickle.dump(labels, f)

    print('{} loading...'.format(rouge_labels_file))
    with open(rouge_labels_file, 'rb') as f:
        rouge_labels = pickle.load(f)

    for t in ['train', 'val', 'test']:
        print('\nPreparing {} data...\n'.format(t))

        file = '../../dataset/{}/splits/data/top50.{}.line'.format(dataset, t)
        df = pd.read_csv(file, sep="\t", names=[
                         "qid", "qidx", "did", "didx", "label"])
        sz = len(df)

        qidxs = []
        didxs = []
        sidxs = []
        labels = []
        distinct_labels = set()

        for qidx in tqdm(set(df['qidx'])):
            for didx, sents_labels in rouge_labels[qidx].items():
                for sidx, label in enumerate(sents_labels):
                    if label in distinct_labels:
                        continue
                    distinct_labels.add(label)

                    qidxs.append(qidx)
                    didxs.append(didx)
                    sidxs.append(sidx)
                    labels.append(label)

        file = './data/{}/top50.{}.rouge'.format(dataset, t)
        df = pd.DataFrame(
            dict(zip(['qidx', 'didx', 'sidx', 'label'], [qidxs, didxs, sidxs, labels])))
        df.to_csv(file, index=None, header=None, sep='\t')
