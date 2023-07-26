from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def load_data_using_mini_batch(dataset: Dataset,mini_batch_size: int,shuffle: bool) -> DataLoader:
    data_loader = DataLoader(dataset,batch_size=mini_batch_size,shuffle=shuffle)
    return data_loader

def make_data_loader_dict(train_data_loader: DataLoader,valid_data_loader: DataLoader) -> dict:
    data_loader_dict = {"train": train_data_loader, "val": valid_data_loader}
    return data_loader_dict