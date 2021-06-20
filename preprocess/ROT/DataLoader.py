from torch.utils.data import Dataset
import pandas as pd
import os


class DatasetCN(Dataset):
    def __init__(self, n, split_set, nrows=None, dataset='weibo'):
        dataset_split_dir = '/data/zhangxueyao/FactChecking/FactCheckingSearch/dataset/'

        if not os.path.exists(dataset_split_dir):
            # Run on 233
            dataset_split_dir = dataset_split_dir.replace('/data/', '/home/')

        if dataset == 'weibo_setting1':
            dataset_split_dir += 'weibo_checked/data/data_splits_setting1/top'
        elif dataset == 'weibo_setting2':
            dataset_split_dir += 'weibo_checked/data/data_splits_setting2/top'
        else:
            assert dataset == 'snopes'
            dataset_split_dir += 'English/_preprocess/data/top'

        self.split_set = split_set
        file = dataset_split_dir + str(n) + '.' + split_set
        if self.split_set == 'rouge.sent':
            self.dataset = pd.read_csv(file, sep="\t", nrows=nrows)
        else:
            self.dataset = pd.read_csv(file, sep="\t", names=["qid", "qidx", "did", "didx", "label"], nrows=nrows)

        print('\n{} loading successfully!\n'.format(file))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.split_set == 'train.line':
            sample = (
                self.dataset.loc[idx, "qid"],
                self.dataset.loc[idx, "qidx"],
                self.dataset.loc[idx, "did"],
                self.dataset.loc[idx, "didx"],
                self.dataset.loc[idx, "label"]
            )
        elif self.split_set in ['val', 'test']:
            sample = (
                self.dataset.loc[idx, "qid"],
                self.dataset.loc[idx, "qidx"],
                # type == list
                eval(self.dataset.loc[idx, "did"]),
                eval(self.dataset.loc[idx, "didx"]),
                eval(self.dataset.loc[idx, "label"])
            )
        elif self.split_set == 'rouge.sent':
            sample = (
                self.dataset.loc[idx, 'qid'],
                self.dataset.loc[idx, "qidx"],
                self.dataset.loc[idx, "did"],
                self.dataset.loc[idx, "didx"],
                self.dataset.loc[idx, "sidx"],
                # type == tuple
                eval(self.dataset.loc[idx, 'label'])
            )
        return sample
