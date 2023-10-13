import time
import torch
import torch.multiprocessing as mp



class MultiProcessWorker(mp.Process):
    def __init__(self, task_queue, result_queue, *args, **kwargs):
        super(MultiProcessWorker, self).__init__(*args, **kwargs)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            func, args, kwargs = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                result = e
            self.result_queue.put(result)



class MultiProcessPool:
    def __init__(self, num_workers=1):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = [MultiProcessWorker(self.task_queue, self.result_queue) for _ in range(num_workers)]
        for w in self.workers:
            w.start()


