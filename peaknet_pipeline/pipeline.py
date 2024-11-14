import torch
import cupy as cp
from cupyx.scipy import ndimage
from contextlib import nullcontext
from peaknet.tensor_transforms import Crop, InstanceNorm, MergeBatchChannelDims

import logging
import time

class Timer:
    def __init__(self, tag=None, is_on=True, cuda_sync=False):
        self.tag = tag
        self.is_on = is_on
        self.cuda_sync = cuda_sync
        self.duration = None

    def __enter__(self):
        if self.is_on:
            if self.cuda_sync:
                torch.cuda.synchronize()
            self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_on:
            if self.cuda_sync:
                torch.cuda.synchronize()
            self.end = time.monotonic()
            self.duration = self.end - self.start
            if self.tag is not None:
                logging.info(f"{self.tag}, duration: {self.duration:.4f} sec.")

class PipelineStage:
    def __init__(self, name, operation, enable_timing=True, cuda_sync=False):
        self.name = name
        self.operation = operation
        self.enable_timing = enable_timing
        self.cuda_sync = cuda_sync
        self.timer = Timer(tag=f"Stage: {name}", is_on=enable_timing, cuda_sync=cuda_sync)

    def process(self, input_data):
        with self.timer:
            return self.operation(input_data)

    @property
    def last_duration(self):
        return self.timer.duration if self.timer.duration is not None else 0.0

class InferencePipeline:
    def __init__(self, model, device, mixed_precision_dtype, H, W, enable_timing=True):
        self.model = model
        self.device = device
        self.mixed_precision_dtype = mixed_precision_dtype
        self.B = None  # TBD
        self.P = None  # TBD
        self.H = H
        self.W = W
        self.structure = cp.ones((3, 3), dtype=cp.float32)
        self.autocast_context = None
        self.setup_autocast()

        # Determine which stages need CUDA synchronization
        is_cuda = 'cuda' in str(self.device)
        self.stages = [
            PipelineStage("data_transfer", self.data_transfer,
                         enable_timing=enable_timing, cuda_sync=is_cuda),
            PipelineStage("preprocess", self.preprocess,
                         enable_timing=enable_timing, cuda_sync=is_cuda),
            PipelineStage("inference", self.inference,
                         enable_timing=enable_timing, cuda_sync=is_cuda),
            PipelineStage("postprocess", self.postprocess,
                         enable_timing=enable_timing, cuda_sync=is_cuda)
        ]

        # Reduce memory
        torch.backends.cuda.matmul.allow_tf32 = True

    def setup_autocast(self):
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        if device_type == 'cpu' or self.mixed_precision_dtype == torch.float32:
            self.autocast_context = nullcontext()
        else:
            self.autocast_context = torch.amp.autocast(device_type=device_type, dtype=self.mixed_precision_dtype)

    def data_transfer(self, batch):
        return batch.to(self.device)

    def preprocess(self, batch):
        if self.B is None:
            self.B, self.P, _, _ = batch.size()
        pad_style = 'top-left'
        transforms = (
            Crop(self.H, self.W, pad_style),  # Bottom-right style padding is supported by default in the underlying crop function if the input size is smaller than the final crop size
            InstanceNorm(),
            MergeBatchChannelDims(),
        )
        for trans in transforms:
            batch = trans(batch)
        return batch

    def inference(self, batch):
        with torch.no_grad():
            with self.autocast_context:
                feature_pred = self.model(batch)
        return feature_pred.softmax(dim=1).argmax(dim=1, keepdim=True)

    def postprocess(self, seg_maps):
        """
        Process segmentation maps to find peak positions for each batch item.

        This function takes a 4D tensor of segmentation maps and processes each map
        to find peak positions. The peaks are then grouped by batch item.

        Args:
            seg_maps (tensor): A 4D tensor of shape (BP, 1, H, W) where:
                B is the batch size,
                P is the number of panels per batch item,
                H is the height of each segmentation map,
                W is the width of each segmentation map.

        Returns:
            list of lists: A list containing B sublists, where B is the batch size.
            Each sublist contains peak positions for a single batch item across all
            its panels. Each peak position is represented as [p, y, x], where:
                p is the panel index (0 to P-1),
                y is the y-coordinate of the peak,
                x is the x-coordinate of the peak.
            The [p, y, x] format is required by downstream software for further processing.
        """
        B = seg_maps.size(0) // self.P  # BP//P
        peak_positions = [[] for _ in range(B)]  # Initialize a list for each batch item
        for idx, seg_map in enumerate(seg_maps.flatten(0,1)):  # (BP,1,H,W)->(BP,H,W), loop over all panels
            seg_map_cp = cp.asarray(seg_map, dtype=cp.float32)
            labeled_map, num_peaks = ndimage.label(seg_map_cp, self.structure)
            peak_coords = ndimage.center_of_mass(seg_map_cp, cp.asarray(labeled_map, dtype=cp.float32), cp.arange(1, num_peaks + 1))
            if len(peak_coords) > 0:
                # Append coordinates for this segmap to the corresponding batch item
                b = idx // self.P  # Batch index
                p = idx % self.P   # Panel index within the batch item
                peak_positions[b].extend([p] + peak.tolist() for peak in peak_coords if len(peak) > 0)
        return peak_positions

    def process_batch(self, batch):
        data = batch
        for stage in self.stages:
            data = stage.process(data)
        return data

    def get_stage_durations(self):
        """Returns a dictionary of stage names and their last recorded durations."""
        return {stage.name: stage.last_duration for stage in self.stages}

    def get_total_duration(self):
        """Returns the total duration of all stages from the last run."""
        return sum(stage.last_duration for stage in self.stages)

    def set_timing_enabled(self, enabled):
        """Enable or disable timing for all stages."""
        for stage in self.stages:
            stage.enable_timing = enabled
            stage.timer.is_on = enabled
