from torch.utils.data import Dataset
from .arxiv_mixin import ArxivMixin

class ArxivDataset(
    Dataset,
    ArxivMixin
):

    def __init__(self, 
        filepath, 
        max_size=100000,
        return_elements=('title', 'abstract')
    ):

        self.return_elements = return_elements

        self.build_dataset(filepath, max_size)
    
    def __getitem__(self, idx):
        dataset_element = self.dataset[idx]
        return tuple(map(dataset_element.get, self.return_elements))

    def __len__(self):
        return len(self.dataset)