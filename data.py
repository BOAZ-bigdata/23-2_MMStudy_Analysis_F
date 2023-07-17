from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def load_data_using_mini_batch(dataset: Dataset,mini_batch_size: int) -> DataLoader:
    data_loader = DataLoader(dataset,batch_size=mini_batch_size,shuffle=True)
    return data_loader
