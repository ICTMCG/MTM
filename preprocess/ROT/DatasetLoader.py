from torch.utils.data import Dataset
import pandas as pd
import os


class DatasetLoader(Dataset):
    def __init__(self, split_set, top=50, nrows=None, dataset='Weibo'):
        dataset_split_dir = '../../dataset/{}/splits/data'.format(dataset)

        self.split_set = split_set
        file = os.path.join(dataset_split_dir,
                            'top{}.{}'.format(top, split_set))
        self.dataset = pd.read_csv(file, sep="\t", names=[
                                   "qid", "qidx", "did", "didx", "label"], nrows=nrows)

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

        return sample
