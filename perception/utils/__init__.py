import time

class Rate:
    def __init__(self, rate):
        self.rate = rate
        self.last_sleep = 0.0 # Long in the future.
        self.time_per_step = 1.0 / float(rate)

    def sleep(self):
        now = time.time()
        time_since_last = now - self.last_sleep
        to_sleep = max(self.time_per_step - time_since_last, 0.0)
        time.sleep(to_sleep)
        self.last_sleep = now
