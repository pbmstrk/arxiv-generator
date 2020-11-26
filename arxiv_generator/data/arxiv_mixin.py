import jsonlines
from tqdm import tqdm


class ArxivMixin:
    def build_dataset(self, filepath, max_size=None, categories=None):
        self.dataset = []
        counter = 0
        with jsonlines.open(filepath) as reader:
            for obj in tqdm(reader):
                if categories and any(cat in obj["categories"] for cat in categories):
                    self.dataset.append(obj)
                    counter += 1
                    if max_size and counter == max_size:
                        break
                if not categories:
                    self.dataset.append(obj)
                    counter += 1
                    if max_size and counter == max_size:
                        break
