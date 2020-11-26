from torch.utils.data import Dataset

from .arxiv_mixin import ArxivMixin


class ArxivDataset(Dataset, ArxivMixin):
    def __init__(
        self,
        filepath,
        max_size=None,
        return_elements=("abstract", "title"),
        categories=None,
    ):

        self.return_elements = return_elements

        self.build_dataset(filepath, max_size, categories)

    def __getitem__(self, idx):
        dataset_element = self.dataset[idx]
        return {key: dataset_element.get(key) for key in self.return_elements}

    def __len__(self):
        return len(self.dataset)
