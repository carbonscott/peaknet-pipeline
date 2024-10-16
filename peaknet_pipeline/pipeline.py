import torch
import cupy as cp
from cupyx.scipy import ndimage
from peaknet.tensor_transforms import Pad, InstanceNorm, PolarCenterCrop, MergeBatchPatchDims

class PipelineStage:
    def __init__(self, name, operation, device):
        self.name = name
        self.operation = operation
        self.stream = torch.cuda.Stream(device=device)
        self.buffer = None

    def process(self, input_data):
        with torch.cuda.stream(self.stream):
            self.buffer = self.operation(input_data)
        return self.buffer

class InferencePipeline:
    def __init__(self, model, device, mixed_precision_dtype):
        self.model = model
        self.device = device
        self.mixed_precision_dtype = mixed_precision_dtype
        self.stages = [
            PipelineStage("data_transfer", self.data_transfer, device),
            PipelineStage("preprocess", self.preprocess, device),
            PipelineStage("inference", self.inference, device),
            PipelineStage("postprocess", self.postprocess, device)
        ]
        self.structure = cp.ones((3, 3), dtype=cp.float32)
        self.autocast_context = None

    def setup_autocast(self):
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        if device_type == 'cpu' or self.mixed_precision_dtype == torch.float32:
            self.autocast_context = nullcontext()
        else:
            self.autocast_context = torch.amp.autocast(device_type=device_type, dtype=self.mixed_precision_dtype)

    def data_transfer(self, batch):
        return batch.to(self.device, non_blocking=True)

    def preprocess(self, batch):
        H_pad, W_pad = 1920, 1920  # TODO: Make these configurable
        pad_style = 'bottom-right'
        transforms = (
            Pad(H_pad, W_pad, pad_style),
            PolarCenterCrop(Hv=1920, Wv=1920, sigma=0.1, num_crop=1),  # TODO: Make these configurable
            MergeBatchPatchDims(),
            InstanceNorm(),
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
        peak_positions = []
        for seg_map in seg_maps:
            seg_map_cp = cp.asarray(seg_map[0], dtype=cp.float32)  # (C,H,W) -> (H,W)
            labeled_map, num_peaks = ndimage.label(seg_map_cp, self.structure)
            peak_coords = ndimage.center_of_mass(seg_map_cp, cp.asarray(labeled_map, dtype=cp.float32), cp.arange(1, num_peaks + 1))
            peak_positions.append(peak_coords)
        return peak_positions

    def convert_to_python_lists(self, peak_positions):
        return [[coord.tolist() for coord in image_peaks] for image_peaks in peak_positions]

    def process_batch(self, batch):
        data = self.stages[0].process(batch)  # data_transfer
        for i in range(1, len(self.stages)):
            data = self.stages[i].process(data)  # preprocess, inference, postprocess
        peak_positions = data
        return self.convert_to_python_lists(peak_positions)

##     def run_inference(self, dataloader):
##         all_peak_positions = []
##         for batch in dataloader:
##             peak_positions = self.process_batch(batch)
##             all_peak_positions.extend(peak_positions)
##         return all_peak_positions
