import torch
import cupy as cp
from cupyx.scipy import ndimage
from contextlib import nullcontext
from peaknet.tensor_transforms import PadAndCrop, InstanceNorm, MergeBatchChannelDims

class PipelineStage:
    def __init__(self, name, operation):
        self.name = name
        self.operation = operation

    def process(self, input_data):
        return self.operation(input_data)

class InferencePipeline:
    def __init__(self, model, device, mixed_precision_dtype, H, W):
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
        self.stages = [
            PipelineStage("data_transfer", self.data_transfer),
            PipelineStage("preprocess", self.preprocess),
            PipelineStage("inference", self.inference),
            PipelineStage("postprocess", self.postprocess)
        ]

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
            PadAndCrop(self.H, self.W, pad_style),
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
