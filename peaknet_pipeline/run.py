import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from datetime import timedelta

from psana_ray.data_reader import DataReader
from psana_ray.shared_queue import create_queue
from peaknet.modeling.convnextv2_bifpn_net import PeakNet, PeakNetConfig, SegHeadConfig
from peaknet.modeling.bifpn_config import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config

from .mpi_utils import init_dist_env
from .data import QueueDataset
from .pipeline import InferencePipeline

from mpi4py import MPI

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

def run_inference(args):
    # Enable distributed env
    init_dist_env()
    uses_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if uses_dist:
        dist_rank       = int(os.environ["RANK"])
        dist_local_rank = int(os.environ["LOCAL_RANK"])
        dist_world_size = int(os.environ["WORLD_SIZE"])
        dist_backend    = args.dist_backend
        dist.init_process_group(backend     = dist_backend,
                                rank        = dist_rank,
                                world_size  = dist_world_size,
                                timeout     = timedelta(seconds = 1800),
                                init_method = "env://",)
        logging.info(f"RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")
    else:
        dist_rank       = 0
        dist_local_rank = 0
        dist_world_size = 1
        logging.info(f"NO distributed environment is required. RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")

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
            # Queue does not exist; create it
            peak_positions_queue = create_queue(queue_name=args.peak_positions_queue_name,
                                                ray_namespace=args.ray_namespace,
                                                maxsize=args.peak_positions_queue_size)
            logging.info(f"Created peak_positions_queue_name: {args.peak_positions_queue_name} in namespace: {args.ray_namespace}")
        # Synchronize all ranks to ensure the queue is created before others try to connect
        comm.Barrier()

        # Non-zero ranks connect to the existing peak_positions_queue
        peak_positions_queue = ray.get_actor(args.peak_positions_queue_name, namespace=args.ray_namespace)
        logging.info(f"Connected to peak_positions_queue_name: {args.peak_positions_queue_name} in namespace: {args.ray_namespace}")

        # Initialize the PeakNetInference
        model = load_model(args.config_path, args.weights_path)
        model.to(device)
        if uses_dist:
            model = DDP(model, device_ids=[dist_local_rank])

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

        # Create and run the inference pipeline
        pipeline = InferencePipeline(model, device, mixed_precision_dtype)
        pipeline.setup_autocast()

        logging.info("InferencePipeline created, starting inference")

        accumulated_results = []
        base_delay = 0.1  # Base delay of 100ms
        max_delay = 2.0   # Maximum delay of 2 seconds
        max_retries = sys.maxsize
        for batch_idx, batch in enumerate(dataloader):
            if batch.numel() == 0:
                logging.warning("Received empty batch, skipping")
                continue

            logging.info(f"Processing batch {batch_idx} of shape: {batch.shape}")
            peak_positions = pipeline.process_batch(batch)

            # Accumulate results for the batch
            batch_results = list(zip(batch.cpu().numpy(), peak_positions))
            accumulated_results.extend(batch_results)

            # Push accumulated results to the new queue when we reach the accumulation step
            if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                retries = 0
                while True:
                    success = ray.get(peak_positions_queue.put.remote(accumulated_results))
                    if success:
                        logging.info(f"Rank {dist_local_rank}, Device {device}: Pushed accumulated results to queue")
                        accumulated_results = []
                        break
                    else:
                        if retries >= max_retries:
                            logging.error(f"Rank {dist_local_rank}, Device {device}: Max retries reached. Unable to push to queue.")
                            break
                        # Use exponential backoff with jitter
                        delay = min(max_delay, base_delay * (2 ** retries))
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter

                        logging.warning(f"Rank {dist_local_rank}, Device {device}: Queue is full, retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                        retries += 1

        # Signal end of data
        if dist_rank == 0:
            ray.get(peak_positions_queue.put.remote(None))
            logging.info("Sent end-of-data signal to peak results queue")

    except KeyboardInterrupt:
        logging.info("Interrupt received, cleaning up...")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
    finally:
        if dataset is not None: dataset.cleanup()
        if uses_dist:
            dist.destroy_process_group()
        ray.shutdown()
        MPI.Finalize()
        logging.info("Inference completed or terminated. Exiting...")

def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Pipeline")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
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
    parser.add_argument("--peak_positions_queue_name", type=str, default="peak_positions_queue",
                        help="Name of the Ray queue to push peak positions")
    parser.add_argument("--peak_positions_queue_size", type=int, default=1000,
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

    run_inference(args)

if __name__ == "__main__":
    main()
