import itertools
import random
from torch.utils import data

class RoundRobin(data.IterableDataset):
    """
    This class will sample iterable datasets in a round robin fashion ad-ininitum.
    When a dataset runs out of juice, it will simply reset it.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_count = len(datasets)

    def __iter__(self):
        datasets = [iter(d) for d in self.datasets]
        i = 0
        while True:
            current_dataset = datasets[i]
            try:
                example = next(current_dataset)
                i = (i + 1) % self.dataset_count
                yield example
            except StopIteration as e:
                datasets[i] = iter(self.datasets[i])
                continue

class Chain(data.IterableDataset):
    def __init__(self, datasets, shuffle=True, infinite=False):
        self.shuffle = shuffle
        self.datasets = datasets
        self.infinite = infinite

    def __iter__(self):
        datasets = self.datasets
        if self.shuffle:
            random.shuffle(datasets)
        if self.infinite:
            for dataset in itertools.cycle(self.datasets):
                for item in dataset:
                    try:
                        yield item
                    except StopIteration:
                        continue
        else:
            for dataset in self.datasets:
                for item in dataset:
                    yield item

    def __len__(self):
        return sum(len(d) for d in self.datasets)

class SamplingPool(data.IterableDataset):
    """
    Maintains a pool of N examples and samples randomly from that pool.
    Useful for mixing different iterable datasets together.
    """
    def __init__(self, dataset, n=1000):
        self.dataset = dataset
        self.n = n

    def __iter__(self):
        pool = []
        iterator = iter(self.dataset)
        for _ in range(self.n):
            try:
                pool.append(next(iterator))
            except StopIteration:
                break

        while True:
            try:
                new_example = next(iterator)
            except StopIteration as e:
                break

            random_index = random.randint(0, len(pool)-1)
            yield pool[random_index]
            pool[random_index] = new_example

        # If the dataset is exhausted, empty the pool.
        while len(pool) > 0:
            random_index = random.randint(0, len(pool)-1)
            yield pool[random_index]
            del pool[random_index]

    def __len__(self):
        return len(self.dataset)

