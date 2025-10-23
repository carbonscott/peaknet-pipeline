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

from contextlib import nullcontext

from .mpi_utils import init_dist_env
from .data import QueueDataset
from .pipeline import InferencePipeline

import torch.multiprocessing as mp
from mpi4py import MPI

import ray
import traceback

TERMINATION_SIGNAL = 'END'

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

    model.eval()

    return model

def barrier_with_timeout(comm, timeout=60):
    """
    Implements a non-blocking barrier with a timeout.

    Args:
        comm: MPI communicator.
        timeout: Maximum time to wait for the barrier in seconds.

    Returns:
        True if the barrier was successfully reached within the timeout, False otherwise.
    """
    request = comm.Ibarrier()
    start_time = time.time()
    while True:
        flag = request.Test()  # All ranks have reached this point???
        if flag:
            return True
        if time.time() - start_time > timeout:
            return False
        time.sleep(1)

def run_inference(args):
    # Enable distributed env
    init_dist_env()
    dist_rank       = int(os.environ.get("RANK", 0))
    dist_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist_world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set up the device
    device = torch.device(f"cuda:{dist_local_rank}")
    torch.cuda.set_device(device)

    dataset = None
    try:
        # Initialize Ray
        ray.init(address=args.ray_address, namespace=args.ray_namespace)

        # Synchronization using MPI
        comm = MPI.COMM_WORLD

        if dist_rank == 0:
            # Try to connect to existing queue first
            try:
                peak_positions_queue = ray.get_actor(args.output_queue_name, namespace=args.ray_namespace)
                logging.info(f"Connected to existing output_queue_name: {args.output_queue_name} in namespace: {args.ray_namespace}")
            except ValueError:  # Ray raises ValueError when actor is not found
                # Queue doesn't exist; create it
                peak_positions_queue = create_queue(queue_name=args.output_queue_name,
                                                    ray_namespace=args.ray_namespace,
                                                    maxsize=args.output_queue_size)
                logging.info(f"Created output_queue_name: {args.output_queue_name} in namespace: {args.ray_namespace}")
        # Synchronize all ranks to ensure the queue is created before others try to connect
        comm.Barrier()

        # Non-zero ranks connect to the existing peak_positions_queue
        peak_positions_queue = ray.get_actor(args.output_queue_name, namespace=args.ray_namespace)
        logging.info(f"Connected to output_queue_name: {args.output_queue_name} in namespace: {args.ray_namespace}")

        # Initialize the PeakNetInference
        model = load_model(args.config_path, args.weights_path)
        model.to(device)

        # Map string dtype to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        mixed_precision_dtype = dtype_map[args.dtype]

        # Compile model if requested - each rank proceeds independently
        if args.compile:
            try:
                logging.info(f"Rank {dist_rank}: Starting model compilation")
                compilation_start = time.time()

                device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
                if device_type == 'cpu' or mixed_precision_dtype == torch.float32:
                    autocast_context = nullcontext()
                else:
                    autocast_context = torch.amp.autocast(device_type=device_type, dtype=mixed_precision_dtype)

                model = torch.compile(
                    model,
                    mode="reduce-overhead",
                    backend="inductor",
                )

                # Simple warmup without barriers
                batch_sizes = args.warmup_batch_sizes
                with torch.no_grad():
                    with autocast_context:
                        for batch_size in batch_sizes:
                            dummy_batch = torch.randn((batch_size, 1, args.H, args.W), device=device, dtype=mixed_precision_dtype)
                            _ = model(dummy_batch)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                compilation_time = time.time() - compilation_start
                logging.info(f"Rank {dist_rank}: Compilation successful, took {compilation_time:.2f} seconds")

                torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Rank {dist_rank}: Compilation failed: {str(e)}")
                logging.warning(f"Rank {dist_rank}: Falling back to uncompiled model")
                logging.error("Traceback:")
                logging.error(traceback.format_exc())

                # Reload the model if compilation failed
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

        # Create and run the inference pipeline
        pipeline = InferencePipeline(model, device, mixed_precision_dtype, args.H, args.W, args.num_overlap)
        ## pipeline.setup_autocast()

        logging.info("InferencePipeline created, starting inference")

        accumulated_results = []
        base_delay = 0.1  # Base delay of 100ms
        max_delay = 2.0   # Maximum delay of 2 seconds
        batch_idx = 0

        for batch_idx, (batch, batch_photon_energy) in enumerate(dataloader):
            if batch.numel() == 0:
                logging.warning("Received empty batch, skipping")
                continue

            logging.info(f"Processing batch {batch_idx} of shape: {batch.shape}")
            peak_positions = pipeline.process_batch(batch, batch_idx)

            # Accumulate results for the batch
            batch_results = list(zip(batch.cpu().numpy(), peak_positions, batch_photon_energy))
            accumulated_results.extend(batch_results)

            # Push accumulated results to the new queue when we reach the accumulation step
            if (batch_idx + 1) % args.accumulation_steps == 0:
                retries = 0
                while True:
                    success = ray.get(peak_positions_queue.put.remote(accumulated_results))
                    if success:
                        logging.info(f"Rank {dist_local_rank}, Device {device}: Pushed accumulated results to queue")
                        accumulated_results = []
                        break
                    else:
                        # Use exponential backoff with jitter
                        delay = min(max_delay, base_delay * (2 ** retries))
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter
                        logging.warning(f"Rank {dist_local_rank}, Device {device}: Queue is full, retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                        if delay < max_delay: retries += 1

        if accumulated_results:
            retries = 0
            while True:
                try:
                    success = ray.get(peak_positions_queue.put.remote(accumulated_results), timeout=5)
                    if success:
                        logging.info(f"Rank {dist_local_rank}, Device {device}: Pushed final accumulated results to queue")
                        break
                    else:
                        # Use exponential backoff with jitter
                        delay = min(max_delay, base_delay * (2 ** retries))
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter
                        logging.warning(f"Rank {dist_local_rank}, Device {device}: Queue is full, retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                        if delay < max_delay: retries += 1
                except ray.exceptions.GetTimeoutError:
                    logging.warning(f"Rank {dist_local_rank}, Device {device}: Timeout while pushing final results, retrying...")

        # Sync progress and signal end of data
        barrier_success = barrier_with_timeout(comm, timeout=300)
        if not barrier_success:
            logging.error("Final Barrier timeout reached. One or more processes may have failed.")
            raise TimeoutError("Final MPI Barrier timed out due to process failure.")
        else:
            logging.info("All processes have reached the final barrier successfully.")

        if dist_rank == 0:
            for _ in range(args.num_consumers):
                try:
                    ray.get(peak_positions_queue.put.remote(TERMINATION_SIGNAL))
                except ray.exceptions.RayActorError as rae:
                    logging.error(f"RayActorError while sending termination signal: {rae}")
                except Exception as ex:
                    logging.error(f"Unexpected error while sending termination signal: {ex}")
            logging.info("Sent end-of-data signal to peak results queue")

    except TimeoutError as te:
        logging.error(f"TimeoutError: {te}")
    except KeyboardInterrupt:
        logging.info("Interrupt received, cleaning up...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
    finally:
        if dataset is not None: dataset.cleanup()
        ray.shutdown()
        MPI.Finalize()
        logging.info("Inference completed or terminated. Exiting...")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Pipeline")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type for mixed precision")
    parser.add_argument("--H", type=int, default=512, help="Height of the image in inference")
    parser.add_argument("--W", type=int, default=512, help="Width of the image in inference")
    parser.add_argument("--num_overlap", type=int, default=3, help="Number of overlap tasks")
    parser.add_argument("--accumulation_steps", type=int, default=10, help="Accumulation step before pushing to queue")
    parser.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use")
    parser.add_argument("--num_consumers", type=int, default=1, help="Number of consumer processes expected.")
    parser.add_argument("--compile", action='store_true', help="Turn on `torch.compile`.")
    parser.add_argument(
        "--warmup_batch_sizes",
        type=int,
        nargs='+',
        default=[4,3,2,1],
        help="Space-separated list of batch sizes to use for warmup (e.g., '4 3 2 1')"
    )
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

    # Use 'spawn' to start fresh worker processes and avoid inheriting parent state, preventing conflicts with MPI and Ray.
    mp.set_start_method('spawn', force=True)

    run_inference(args)

if __name__ == "__main__":
    main()
