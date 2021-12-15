from torch.utils.data import Dataset
import pandas as pd
import os


class DatasetLoader(Dataset):
    def __init__(self, split_set, top=50, nrows=None, dataset='Weibo'):
        dataset_split_dir = '../dataset/{}/splits/data'.format(dataset)

        self.split_set = split_set
        file = os.path.join(dataset_split_dir,
                            'top{}.{}'.format(top, split_set))

        """
        qid: the unique id of the query claim
        qidx: the index of the query claim
        did: the unique id(s) of the debunking article(s)
        didx: the index(es) of the debunking article(s)
        label: the label(s) of the "query, article(s)" pair(s), 1 for "relevant" and 0 for "irrelevant"
        """

        self.dataset = pd.read_csv(file, sep="\t", names=[
                                   'qid', 'qidx', 'did', 'didx', 'label'], nrows=nrows)

        # Class Distribution
        if '.line' in split_set:
            labels = set(self.dataset['label'])
            class_nums = dict()
            for l in labels:
                class_nums[l] = len(self.dataset[self.dataset['label'] == l])
                print('label {}: sz = {}'.format(l, class_nums[l]))

            sz = len(self.dataset)
            self.samples_weights = [sz/class_nums[l]
                                    for l in self.dataset['label']]

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
        else:
            sample = (
                self.dataset.loc[idx, "qid"],
                self.dataset.loc[idx, "qidx"],
                # type == list
                eval(self.dataset.loc[idx, "did"]),
                eval(self.dataset.loc[idx, "didx"]),
                eval(self.dataset.loc[idx, "label"])
            )

        return sample
