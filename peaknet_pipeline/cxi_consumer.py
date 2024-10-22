import argparse
import logging
import signal
import sys
import time
import random
import h5py
import numpy as np
from mpi4py import MPI
import ray
import os
import traceback
from crystfel_stream_parser.joblib_engine import StreamParser
from crystfel_stream_parser.cheetah_converter import CheetahConverter

from mpi4py import MPI

TERMINATION_SIGNAL = 'END'

def setup_logging(log_level):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def signal_handler(sig, frame):
    """Handle termination signals for graceful shutdown."""
    logging.info("Termination signal received. Shutting down consumer...")
    ray.shutdown()
    MPI.Finalize()
    sys.exit(0)

def write_cxi_file(rank, images, peak_positions, output_dir, basename, chunk_id, max_num_peak, encoder_value, photon_energy, cheetah_converter):
    """
    Write a CXI file with the given images and peak positions.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{basename}_rank{rank:d}_chunk{chunk_id:d}.cxi"
    filepath = os.path.join(output_dir, filename)

    try:
        with h5py.File(filepath, 'w') as f:
            num_events = len(images)
            image_shape = cheetah_converter.convert_to_cheetah_img(images[0]).shape  # Convert to Cheetah image shape

            # Create datasets
            f.create_dataset('/entry_1/data_1/data', (num_events, *image_shape), dtype='float32')
            f.create_dataset('/entry_1/result_1/peakXPosRaw', (num_events, max_num_peak), dtype='float32')
            f.create_dataset('/entry_1/result_1/peakYPosRaw', (num_events, max_num_peak), dtype='float32')
            f.create_dataset('/entry_1/result_1/nPeaks', (num_events,), dtype='int')
            f.create_dataset('/entry_1/result_1/peakTotalIntensity', (num_events, max_num_peak), dtype = 'float32')  # Currently a placeholder, no real intensity value will be filled in

            # Create LCLS datasets
            f.create_dataset('/LCLS/detector_1/EncoderValue', (1,), dtype='float32', data=[encoder_value])
            f.create_dataset('/LCLS/photon_energy_eV', (1,), dtype='float32', data=[photon_energy])

            # Write data
            for event_enum_idx, (image, peaks) in enumerate(zip(images, peak_positions)):
                cheetah_image = cheetah_converter.convert_to_cheetah_img(image)
                f['/entry_1/data_1/data'][event_enum_idx] = cheetah_image

                cheetah_peaks = cheetah_converter.convert_to_cheetah_coords(peaks)
                num_peaks = len(cheetah_peaks)
                f['/entry_1/result_1/nPeaks'][event_enum_idx] = num_peaks

                for peak_enum_idx, (_, cheetahRow, cheetahCol) in enumerate(cheetah_peaks):
                    if peak_enum_idx >= max_num_peak:
                        break
                    f['/entry_1/result_1/peakYPosRaw'][event_enum_idx, peak_enum_idx] = cheetahRow
                    f['/entry_1/result_1/peakXPosRaw'][event_enum_idx, peak_enum_idx] = cheetahCol

        logging.info(f"Wrote CXI file: {filepath}")
    except Exception as e:
        logging.error(f"Error writing CXI file {filepath}:")
        logging.error(traceback.format_exc())
        raise

def consume_and_write(queue_name, ray_namespace, rank, output_dir, basename, save_every, max_num_peak, min_num_peak, encoder_value, photon_energy, cheetah_converter):
    """
    Consume peak positions from the specified Ray queue and write to CXI files.
    """
    comm = MPI.COMM_WORLD
    terminate = False

    try:
        peak_positions_queue = ray.get_actor(queue_name, namespace=ray_namespace)
        logging.info(f"Rank {rank}: Connected to queue '{queue_name}' in namespace '{ray_namespace}'.")
    except Exception as e:
        logging.error(f"Rank {rank}: Failed to get queue actor '{queue_name}':")
        logging.error(traceback.format_exc())
        return

    base_delay = 0.1
    max_delay = 5.0
    retries = 0
    iteration = 0
    chunk_id = 0
    accumulated_images = []
    accumulated_peak_positions = []

    while not terminate:
        # Check if any rank has received the termination signal
        terminate = comm.bcast(terminate, root=0)
        if terminate:
            logging.info(f"Rank {rank}: Received broadcast termination signal.")
            break

        try:
            data = ray.get(peak_positions_queue.get.remote())

            if data == TERMINATION_SIGNAL:
                logging.info(f"Rank {rank}: Received sentinel. No more data to consume.")
                terminate = True
                terminate = comm.bcast(terminate, root=rank)
                break

            if data is None:
                delay = min(max_delay, base_delay * (2 ** retries))
                jitter = random.uniform(0, 0.1 * delay)
                total_delay = delay + jitter
                logging.warning(f"Rank {rank}: Queue is empty, retrying in {total_delay:.2f} seconds...")
                time.sleep(total_delay)
                if delay < max_delay:
                    retries += 1
                continue

            retries = 0

            # TODO: Need better logging... otherwise it feels like the queue is always empty
            logging.info(f"Rank {rank}: Pulled data from Queue...")

            for image, peak_positions in data:
                if len(peak_positions) >= min_num_peak:
                    accumulated_images.append(image)
                    accumulated_peak_positions.append(peak_positions)

            iteration += 1
            if iteration % save_every == 0 and accumulated_images:
                write_cxi_file(rank, accumulated_images, accumulated_peak_positions, output_dir, basename, chunk_id, max_num_peak, encoder_value, photon_energy, cheetah_converter)
                chunk_id += 1
                accumulated_images = []
                accumulated_peak_positions = []
                logging.info(f"Rank {rank}: Wrote CXI file at iteration {iteration}.")

        except ray.exceptions.RayActorError as e:
            logging.error(f"Rank {rank}: Queue actor is dead:")
            ## logging.error(traceback.format_exc())
            terminate = True
            terminate = comm.bcast(terminate, root=rank)
            break
        except Exception as e:
            logging.error(f"Rank {rank}: Error while consuming data:")
            logging.error(traceback.format_exc())
            time.sleep(base_delay)

    # Write any remaining data
    if accumulated_images:
        try:
            write_cxi_file(rank, accumulated_images, accumulated_peak_positions, output_dir, basename, chunk_id, max_num_peak, encoder_value, photon_energy, cheetah_converter)
            logging.info(f"Rank {rank}: Wrote final CXI file.")
        except Exception as e:
            logging.error(f"Rank {rank}: Error writing final CXI file:")
            logging.error(traceback.format_exc())

    logging.info(f"Rank {rank}: Exiting.")

def main():
    parser = argparse.ArgumentParser(description="PeakNet Pipeline Consumer with CXI Writer")
    parser.add_argument("--queue_name", type=str, default="peak_positions_queue",
                        help="Name of the Ray queue to consume from")
    parser.add_argument("--ray_namespace", type=str, default="my",
                        help="Ray namespace where the queue resides")
    parser.add_argument("--ray_address", type=str, default="auto",
                        help="Address of the Ray cluster")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save CXI files")
    parser.add_argument("--basename", type=str, default="peaknet_output",
                        help="Base name for output CXI files")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Number of iterations before writing a new CXI file")
    parser.add_argument("--max_num_peak", type=int, default=2048,
                        help="Maximum number of peaks per event")
    parser.add_argument("--min_num_peak", type=int, default=10,
                        help="Minimum number of peaks required to save an event")
    parser.add_argument("--encoder_value", type=float, default=0.0,
                        help="Encoder value for LCLS dataset")
    parser.add_argument("--photon_energy", type=float, default=0.0,
                        help="Photon energy for LCLS dataset")
    parser.add_argument("--geom_file", type=str, required=True,
                        help="Path to the geometry file for CheetahConverter")
    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Setup logging
    setup_logging(args.log_level)

    # Handle termination signals for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize Ray
    try:
        ray.init(address=args.ray_address, namespace=args.ray_namespace, ignore_reinit_error=True)
        logging.info(f"Rank {rank}: Ray initialized successfully.")
    except Exception as e:
        logging.error(f"Rank {rank}: Failed to initialize Ray:")
        logging.error(traceback.format_exc())
        MPI.Finalize()
        sys.exit(1)

    # Initialize CheetahConverter
    try:
        geom_block = StreamParser(args.geom_file).parse(num_cpus=1, returns_stream_dict=True)[0].get('GEOM_BLOCK')  # List of Dict
        cheetah_converter = CheetahConverter(geom_block)
        logging.info(f"Rank {rank}: CheetahConverter initialized successfully.")
    except Exception as e:
        logging.error(f"Rank {rank}: Failed to initialize CheetahConverter:")
        logging.error(traceback.format_exc())
        MPI.Finalize()
        sys.exit(1)

    # Start consuming data and writing CXI files
    consume_and_write(
        queue_name=args.queue_name,
        ray_namespace=args.ray_namespace,
        rank=rank,
        output_dir=args.output_dir,
        basename=args.basename,
        save_every=args.save_every,
        max_num_peak=args.max_num_peak,
        min_num_peak=args.min_num_peak,
        encoder_value=args.encoder_value,
        photon_energy=args.photon_energy,
        cheetah_converter=cheetah_converter
    )

    # Shutdown Ray after consumption
    ray.shutdown()
    logging.info(f"Rank {rank}: Ray shutdown completed.")

    # Finalize MPI
    MPI.Finalize()

if __name__ == "__main__":
    main()
