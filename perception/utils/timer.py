import time
import numpy as np
from datetime import datetime

class Timing:
    def __init__(self):
        self.times = {}
        self.finished = {}

    def start(self, tag):
        assert tag not in self.times, f"{tag} already started"
        start = datetime.now()
        self.times[tag] = start

    def end(self, tag):
        start = self.times[tag]
        values = self.finished.get(tag, [])
        diff = datetime.now() - start
        values.append(diff.total_seconds())
        self.finished[tag] = values
        del self.times[tag]

    def print(self):
        space = " " * 9
        header = f"|\ttag{space}\t|\tavg\t|\tvar\t|"
        print(header)
        print("-" * (len(header) + 16))
        for tag, values in self.finished.items():
            avg = np.mean(values)
            name = tag[:12] + " " * (12 - len(tag))
            std = np.std(values)
            print(f"|\t{name}\t|\t{avg}\t|\t{std}\t|")

