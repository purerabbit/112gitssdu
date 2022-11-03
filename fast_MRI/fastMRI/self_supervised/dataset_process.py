from torch.utils import data as Data
from typing import Sequence, List, Union
import torch
import numpy as np

# def arbitrary_dataset_split(dataset: Data.Dataset,
#                             indices_list: Sequence[Sequence[int]]
#                             ) -> List[torch.utils.data.Subset]:
#     return [Data.Subset(dataset, indices) for indices in indices_list]

def arbitrary_dataset_split(dataset: Data.Dataset,
                            train_nums:int,
                            val_nums:int,
                            test_nums:int
                            ):
    train_indices=np.arange(0,train_nums)
    val_indices=np.arange(train_nums,train_nums+val_nums)
    test_indices=np.arange(train_nums+val_nums, train_nums+val_nums+test_nums)
    indices_list=[train_indices, val_indices, test_indices]
    return [Data.Subset(dataset, indices) for indices in indices_list]


# def datasets2loaders(datasets: Sequence[Data.Dataset],
#                      *,
#                      batch_size: Sequence[int] = (1, 1, 1),  # train, val, test
#                      is_shuffle: Sequence[bool] = (True, False, False),  # train, val, test
#                      num_workers: int = 0) -> Sequence[Data.DataLoader]:
#     """
#     a tool for build N-datasets into N-loaders
#     """
#     assert isinstance(datasets[0], Data.Dataset)
#     n_loaders = len(datasets)
#     assert n_loaders == len(batch_size)
#     assert n_loaders == len(is_shuffle)

#     loaders = []
#     for i in range(n_loaders):
#         loaders.append(
#             Data.DataLoader(datasets[i], batch_size=batch_size[i], shuffle=is_shuffle[i], num_workers=num_workers)
#         )

#     return loaders