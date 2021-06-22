import pickle
import json
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
import os


def get_tokens(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False)


if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenize by Transoformers')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pretrained_model', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    pretrained_model = args.pretrained_model
    save_dir = 'data/{}'.format(dataset)

    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    if dataset == 'Weibo':
        with open('../../dataset/Weibo/raw/FN_11934_filtered.json', 'r') as f:
            FN = json.load(f)
        with open('../../dataset/Weibo/raw/DN_27505_filtered.json', 'r') as f:
            DN = json.load(f)
    else:
        pass

    fn_tokens = [get_tokens(fn['content_all']) for fn in tqdm(FN)]

    dn_tokens = []
    for dn in tqdm(DN):
        tokens = [get_tokens(sent) for sent in dn['content_all']]
        dn_tokens.append(tokens)

    fn_df = pd.DataFrame({'tokens_num': [len(tokens) for tokens in fn_tokens]})
    dn_df = pd.DataFrame(
        {'tokens_num': [len(tokens) for sents in dn_tokens for tokens in sents]})
    dn_sents_df = pd.DataFrame(
        {'sents_num': [len(sents) for sents in dn_tokens]})

    print('Dataset: {}, Pretrained Model: {}\n'.format(dataset, pretrained_model))
    print('Claim: {}\nTokens num: {}\n'.format(len(fn_df), fn_df.describe()))
    print('Articles: {}, Sentences: {}'.format(len(dn_sents_df), len(dn_df)))
    print('Articles\' Sentences num: {}'.format(dn_sents_df.describe()))
    print('Sentences\' Tokens num: {}'.format(dn_df.describe()))

    with open(os.path.join(save_dir, 'FN_{}.pkl'.format(pretrained_model)), 'wb') as f:
        pickle.dump(fn_tokens, f)
    with open(os.path.join(save_dir, 'DN_{}.pkl'.format(pretrained_model)), 'wb') as f:
        pickle.dump(dn_tokens, f)

    fn_df.describe().to_csv(os.path.join(save_dir, 'FN_{}.csv'.format(pretrained_model)))
    dn_df.describe().to_csv(os.path.join(save_dir, 'DN_{}.csv'.format(pretrained_model)))
    dn_sents_df.describe().to_csv(os.path.join(
        save_dir, 'DN_sents_num_{}.csv'.format(pretrained_model)))
