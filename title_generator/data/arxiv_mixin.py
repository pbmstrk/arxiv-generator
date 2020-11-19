import jsonlines
from tqdm import tqdm

class ArxivMixin:

    def build_dataset(self, filepath, max_size):
        self.dataset = []
        counter = 0
        with jsonlines.open(filepath) as reader:
            for obj in tqdm(reader):
                self.dataset.append(obj)
                counter += 1
                if counter == max_size:
                    break
