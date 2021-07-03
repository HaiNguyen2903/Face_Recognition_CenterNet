# from re import split
# import numpy as np
# from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate
# from torch.utils.data.sampler import SubsetRandomSampler
# from data.wider_face import *
# from IPython import embed


# class BaseDataLoader(DataLoader):
#     """
#     Base class for all data loaders
#     """
#     def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
#         self.validation_split = validation_split
#         self.shuffle = shuffle
#         self.batch_size = batch_size

#         self.batch_idx = 0
#         self.n_samples = len(dataset)

#         self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

#         self.init_kwargs = {
#             'dataset': dataset,
#             'batch_size': batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': collate_fn,
#             'num_workers': num_workers
#         }
#         super().__init__(sampler=self.sampler, **self.init_kwargs)

#     def _split_sampler(self, split):
#         if split == 0.0:
#             return None, None

#         idx_full = np.arange(self.n_samples)

#         np.random.seed(0)
#         np.random.shuffle(idx_full)

#         if isinstance(split, int):
#             assert split > 0
#             assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
#             len_valid = split
#         else:
#             len_valid = int(self.n_samples * split)

#         valid_idx = idx_full[0:len_valid]
#         train_idx = np.delete(idx_full, np.arange(0, len_valid))

#         train_sampler = SubsetRandomSampler(train_idx)
#         valid_sampler = SubsetRandomSampler(valid_idx)

#         # turn off shuffle option which is mutually exclusive with sampler
#         self.shuffle = False
#         self.n_samples = len(train_idx)

#         return train_sampler, valid_sampler

#     def split_validation(self):
#         if self.valid_sampler is None:
#             return None
#         else:
#             return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

# class DataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """

#     def __init__(self, split, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         opt = None
#         self.dataset = WiderFaceDataset(split)

#         # if sampler is not None:
#         #     self.sampler = sampler
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


# class CustomDataLoader(DataLoader):
#     def __init__(self, dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE,
#                 num_workers=NUMBER_WORKER, collate_fn=default_collate, sampler=None):

#         self.n_samples = len(dataset)

#         super().__init__(dataset, batch_size, shuffle,
#                         num_workers, collate_fn, sampler)

# embed()



from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from IPython import embed
from data.wider_face import *
import numpy as np

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, validation_split=0.0, num_workers=NUMBER_WORKER, collate_fn=default_collate, sampler=None):
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        self.sampler = sampler

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def train_test_split(self):
        split = self.validation_split

        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)


        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        print('train sampler:', len(train_sampler))
        print('val sampler:', len(valid_sampler))


        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        train_loader = CustomDataLoader(dataset=self.dataset, batch_size=self.batch_size, 
                                        shuffle=self.shuffle, sampler=train_sampler)
        valid_loader = CustomDataLoader(dataset=self.dataset, batch_size=self.batch_size, 
                                        shuffle=self.shuffle, sampler=valid_sampler)

        return train_loader, valid_loader


# embed(header='Debugging')

# data = WiderFaceDataset(split='train')
# loader = CustomDataLoader(dataset=data, validation_split=0.1)

# train_loader, val_loader = loader.train_test_split()