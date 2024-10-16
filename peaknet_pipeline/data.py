import torch
import logging
import time
from psana_ray.data_reader import DataReader, DataReaderError
from torch.utils.data import IterableDataset

class QueueDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.worker_readers = {}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            worker_id = 0
            num_workers = 1
        else:  # in a worker process
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if worker_id not in self.worker_readers:
            self.worker_readers[worker_id] = DataReader()
            self.worker_readers[worker_id].connect()
            logging.debug(f"Worker {worker_id}: Created and connected new DataReader")

        return self.data_iterator(worker_id, num_workers)

    def data_iterator(self, worker_id, num_workers):
        reader = self.worker_readers[worker_id]
        while True:
            try:
                data = reader.read()
                if data is None:
                    logging.debug(f"Worker {worker_id}: No data received, sleeping...")
                    time.sleep(0.1)  # Short sleep to avoid busy-waiting
                    continue
                rank, idx, image_data = data
                if idx % num_workers == worker_id:
                    logging.debug(f"Worker {worker_id}: Received data: rank={rank}, idx={idx}, image_shape={image_data.shape}")
                    tensor = torch.tensor(image_data).unsqueeze(0)  # (H,W) -> (1,H,W)
                    yield tensor
            except DataReaderError as e:
                logging.error(f"Worker {worker_id}: DataReader error: {e}")
                break
            except Exception as e:
                logging.error(f"Worker {worker_id}: Unexpected error in QueueDataset: {e}")
                time.sleep(1)  # Longer sleep on unexpected errors

    def cleanup(self):
        """Explicit cleanup method to close all DataReaders."""
        for worker_id, reader in self.worker_readers.items():
            try:
                reader.close()
                logging.debug(f"Worker {worker_id}: DataReader closed in cleanup")
            except Exception as e:
                logging.error(f"Worker {worker_id}: Error closing DataReader in cleanup: {e}")
        self.worker_readers.clear()

    def __del__(self):
        self.cleanup()
