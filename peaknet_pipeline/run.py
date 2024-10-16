import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from datetime import timedelta

from psana_ray.data_reader import DataReader
from peaknet.modeling.convnextv2_bifpn_net import PeakNet, PeakNetConfig, SegHeadConfig
from peaknet.modeling.bifpn_config import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config

from .mpi_utils import init_dist_env
from .data import QueueDataset
from .pipeline import InferencePipeline

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

    model =  PeakNet(peaknet_config)

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
        # Initialize the PeakNetInference
        model = load_model(args.config_path, args.weights_path)
        model.to(device)
        if uses_dist:
            model = DDP(model, device_ids=[dist_local_rank])

        # Set up the data reader and dataset
        dataset = QueueDataset()

        # Set up the dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            ## prefetch_factor=2,
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

        all_peak_positions = []
        for batch in dataloader:
            if batch.numel() == 0:
                logging.warning("Received empty batch, skipping")
                continue

            logging.info(f"Processing batch of shape: {batch.shape}")
            peak_positions = pipeline.process_batch(batch)
            all_peak_positions.extend(peak_positions)
            logging.info(f"Rank {dist_local_rank}, Device {device}: Processed batch, found {tuple(len(peaks) for peaks in peak_positions)} peaks")

        logging.info(f"Rank {dist_local_rank}, Device {device}: Processed all batches, found {tuple(len(peaks) for peaks in all_peak_positions)} peaks in total")

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
        logging.info("Inference completed or terminated. Exiting...")

def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Pipeline")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Data type for mixed precision")
    parser.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"],
                        help="Distributed backend to use")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    run_inference(args)

if __name__ == "__main__":
    main()
