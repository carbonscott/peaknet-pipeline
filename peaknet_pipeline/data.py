import torch
from torch.utils.data import IterableDataset
from psana_ray.data_reader import DataReader, DataReaderError
import logging
from typing import Optional

class QueueDataset(IterableDataset):
    def __init__(self, queue_name: str = "shared_queue", ray_namespace: str = 'my'):
        super().__init__()
        self.queue_name = queue_name
        self.ray_namespace = ray_namespace
        self.reader: Optional[DataReader] = None
        logging.info(f"QueueDataset initialized with queue_name={queue_name}, ray_namespace={ray_namespace}")

    def __iter__(self):
        if self.reader is None:
            try:
                self.reader = DataReader(queue_name=self.queue_name, ray_namespace=self.ray_namespace)
                self.reader.connect()
                logging.info("Created and connected new DataReader")
            except Exception as e:
                logging.error(f"Failed to create or connect DataReader: {e}")
                raise
        return self

    def __next__(self):
        if self.reader is None:
            raise RuntimeError("DataReader not initialized. Make sure to iterate over the dataset.")

        try:
            data = self.reader.read()
            if data is None:
                logging.debug("No data received, stopping iteration")
                raise StopIteration

            rank, idx, image_data = data
            logging.info(f"Received data: rank={rank}, idx={idx}, image_shape={image_data.shape}")
            tensor = torch.tensor(image_data).unsqueeze(0)
            return tensor

        except DataReaderError as e:
            logging.error(f"DataReader error: {e}")
            raise StopIteration
        except Exception as e:
            logging.error(f"Unexpected error in QueueDataset: {e}")
            raise

    def cleanup(self):
        if self.reader:
            try:
                self.reader.close()
                logging.info("DataReader closed in cleanup")
            except Exception as e:
                logging.error(f"Error closing DataReader in cleanup: {e}")
            self.reader = None

    def __del__(self):
        self.cleanup()
