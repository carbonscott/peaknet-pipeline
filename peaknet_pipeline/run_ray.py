import os
import sys
import random
import time
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
import logging

from psana_ray.data_reader import DataReader
from psana_ray.shared_queue import create_queue
from peaknet.modeling.convnextv2_bifpn_net import PeakNet, PeakNetConfig, SegHeadConfig
from peaknet.modeling.bifpn_config import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config

from .data import QueueDataset
from .pipeline import InferencePipeline

import torch.multiprocessing as mp

import ray
import traceback

def load_model(config_path, weights_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Config
    model_params              = config.get("model")
    backbone_params           = model_params.get("backbone")
    hf_model_config           = backbone_params.get("hf_config")
    bifpn_params              = model_params.get("bifpn")
    bifpn_block_params        = bifpn_params.get("block")
    bifpn_block_bn_params     = bifpn_block_params.get("bn")
    bifpn_block_fusion_params = bifpn_block_params.get("fusion")
    seghead_params            = model_params.get("seg_head")

    # Backbone
    backbone_config = ConvNextV2Config(**hf_model_config)

    # BiFPN
    bifpn_block_params["bn"]     = BNConfig(**bifpn_block_bn_params)
    bifpn_block_params["fusion"] = FusionConfig(**bifpn_block_fusion_params)
    bifpn_params["block"]        = BiFPNBlockConfig(**bifpn_block_params)
    bifpn_config                 = BiFPNConfig(**bifpn_params)

    # Seg head
    seghead_config = SegHeadConfig(**seghead_params)

    # PeakNet
    peaknet_config = PeakNetConfig(
        backbone = backbone_config,
        bifpn    = bifpn_config,
        seg_head = seghead_config,
    )

    model = PeakNet(peaknet_config)

    # Load weights
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)

    return model


@ray.remote(num_gpus=1)
def inference_task(args):
    # Set up the device
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = load_model(args.config_path, args.weights_path)
    model.to(device)

    # Set up the data reader and dataset
    dataset = QueueDataset(queue_name=args.input_queue_name, ray_namespace=args.ray_namespace)

    # Set up the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Map string dtype to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    mixed_precision_dtype = dtype_map[args.dtype]

    # Create the inference pipeline
    pipeline = InferencePipeline(model, device, mixed_precision_dtype)
    pipeline.setup_autocast()

    logging.info(f"InferencePipeline created on device {device}, starting inference")

    # Get the output queue
    peak_positions_queue = ray.get_actor(args.output_queue_name, namespace=args.ray_namespace)

    accumulated_results = []
    base_delay = 0.1
    max_delay = 2.0
    batch_idx = 0
    try:
        for batch in dataloader:
            if batch.numel() == 0:
                logging.warning("Received empty batch, skipping")
                continue

            logging.info(f"Processing batch {batch_idx} of shape: {batch.shape}")
            peak_positions = pipeline.process_batch(batch)

            # Accumulate results for the batch
            batch_results = list(zip(batch.cpu().numpy(), peak_positions))
            accumulated_results.extend(batch_results)

            # Push accumulated results to the queue when we reach the accumulation step
            if (batch_idx + 1) % args.accumulation_steps == 0:
                retries = 0
                while True:
                    success = ray.get(peak_positions_queue.put.remote(accumulated_results))
                    if success:
                        logging.info(f"Device {device}: Pushed accumulated results to queue")
                        accumulated_results = []
                        break
                    else:
                        delay = min(max_delay, base_delay * (2 ** retries))
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter
                        logging.warning(f"Device {device}: Queue is full, retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                        if delay < max_delay: retries += 1
            batch_idx += 1

        if accumulated_results:
            retries = 0
            while True:
                try:
                    success = ray.get(peak_positions_queue.put.remote(accumulated_results), timeout=5)
                    if success:
                        logging.info(f"Device {device}: Pushed final accumulated results to queue")
                        break
                    else:
                        delay = min(max_delay, base_delay * (2 ** retries))
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter
                        logging.warning(f"Device {device}: Queue is full, retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                        if delay < max_delay: retries += 1
                except ray.exceptions.GetTimeoutError:
                    logging.warning(f"Device {device}: Timeout while pushing final results, retrying...")

    except Exception as e:
        logging.error(f"An error occurred in inference task: {str(e)}")
    finally:
        dataset.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Pipeline")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type for mixed precision")
    parser.add_argument("--accumulation_steps", type=int, default=10, help="Accumulation step before pushing to queue")
    parser.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")

    # New arguments for second queue
    parser.add_argument("--output_queue_name", type=str, default="peak_positions_queue",
                        help="Name of the Ray queue to push peak positions")
    parser.add_argument("--output_queue_size", type=int, default=1000,
                        help="Maximum size for the peak_positions_queue")

    # Arguments for the first queue
    parser.add_argument("--input_queue_name", type=str, default="input",
                        help="Name of the Ray queue to pull raw data from")

    # Ray cluster address and namespace
    parser.add_argument("--ray_address", type=str, default="auto",
                        help="Address of the Ray cluster")
    parser.add_argument("--ray_namespace", type=str, default="my",
                        help="Ray namespace to use for both queues")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    ## # Use 'spawn' to start fresh worker processes and avoid inheriting parent state, preventing conflicts with MPI and Ray.
    ## mp.set_start_method('spawn', force=True)

    try:
        # Initialize Ray
        ray.init(address=args.ray_address, namespace=args.ray_namespace, log_to_driver=True)

        # Create the output queue (only if it doesn't exist)
        try:
            peak_positions_queue = ray.get_actor(args.output_queue_name, namespace=args.ray_namespace)
        except ValueError:
            peak_positions_queue = create_queue(queue_name=args.output_queue_name,
                                                ray_namespace=args.ray_namespace,
                                                maxsize=args.output_queue_size)
            logging.info(f"Created output_queue_name: {args.output_queue_name} in namespace: {args.ray_namespace}")

        # Start inference tasks
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        tasks = [inference_task.remote(args) for _ in range(num_gpus)]

        # Wait for all tasks to complete
        ray.get(tasks)

        # Signal end of data
        ray.get(peak_positions_queue.put.remote(None))
        logging.info("Sent end-of-data signal to peak results queue")

        ray.shutdown()
        logging.info("Inference completed. Exiting...")

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up Ray
        if ray.is_initialized():
            ray.shutdown()
            logging.info("Ray has been shut down.")
        logging.info("Inference completed or terminated. Exiting...")

if __name__ == "__main__":
    main()
