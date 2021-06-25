from torch.utils.data import Dataset
import pandas as pd
import os


class DatasetLoader(Dataset):
    def __init__(self, split_set, top=50, nrows=None, dataset='Weibo'):
        dataset_split_dir = './data/{}'.format(dataset)

        self.split_set = split_set
        file = os.path.join(dataset_split_dir,
                            'top{}.{}.rouge'.format(top, split_set))
        self.dataset = pd.read_csv(file, sep="\t", names=[
                                   "qidx", "didx", "sidx", "label"], nrows=nrows)

        print('\n{} loading successfully!\n'.format(file))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = (
            self.dataset.loc[idx, "qidx"],
            self.dataset.loc[idx, "didx"],
            self.dataset.loc[idx, "sidx"],
            # type == tuple
            eval(self.dataset.loc[idx, "label"])
        )
        return sample
