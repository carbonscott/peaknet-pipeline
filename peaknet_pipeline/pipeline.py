import torch
import cupy as cp
from cupyx.scipy import ndimage
from contextlib import nullcontext
from peaknet.tensor_transforms import Crop, InstanceNorm, MergeBatchChannelDims

import torch.cuda.nvtx as nvtx

class InferencePipeline:
    def __init__(self,
                 model,
                 device,
                 mixed_precision_dtype,
                 H, W,
                 num_overlap=2):
        """
        Args:
            model (torch.nn.Module): Your PyTorch model (already on device).
            device (torch.device or str): e.g. 'cuda:0'.
            mixed_precision_dtype (torch.dtype): e.g. torch.float16.
            H, W (int): Dimensions for cropping.
            num_overlap (int): Size of ring buffer for concurrency.
        """
        self.model = model
        self.device = device
        self.mixed_precision_dtype = mixed_precision_dtype
        self.H = H
        self.W = W
        self.B = None
        self.P = None

        # Create GPU streams for each stage
        self.stage1_h2d_stream = torch.cuda.Stream()
        self.stage2_preprocess_stream = torch.cuda.Stream()
        self.stage3_inference_stream = torch.cuda.Stream()
        self.stage4_postprocess_stream = torch.cuda.Stream()

        # Keep ring buffers for intermediate outputs
        self.num_overlap = num_overlap
        self.preproc_outputs = [None] * num_overlap
        self.inference_outputs = [None] * num_overlap
        self.postprocess_outputs = [None] * num_overlap

        # Pre-build a small struct for cupy ops
        self.structure = cp.ones((3, 3), dtype=cp.float32)

        # Set up autocast context
        if "cuda" in str(device) and mixed_precision_dtype != torch.float32:
            self.autocast_context = torch.amp.autocast(device_type='cuda',
                                                       dtype=mixed_precision_dtype)
        else:
            self.autocast_context = nullcontext()

        # Optional: enable TF32 for matmul on Ampere+ GPUs
        ## torch.backends.cuda.matmul.allow_tf32 = True

    def stage1_h2d(self, batch_cpu):
        """Stage 1: Transfer CPU -> GPU."""
        return batch_cpu.to(self.device, non_blocking=True)

    def stage2_preprocess(self, batch_gpu):
        """Stage 2: Crop, InstanceNorm, MergeBatchChannelDims."""
        if self.B is None:
            self.B, self.P, _, _ = batch_gpu.shape

        transforms = (
            Crop(self.H, self.W, "top-left"),
            InstanceNorm(),
            MergeBatchChannelDims(),
        )
        for trans in transforms:
            batch_gpu = trans(batch_gpu)
        return batch_gpu

    def stage3_inference(self, preproc_out):
        """Stage 3: Inference with optional AMP autocast."""
        with torch.no_grad():
            with self.autocast_context:
                logits = self.model(preproc_out)
        return logits.softmax(dim=1).argmax(dim=1, keepdim=True)

    def stage4_postprocess(self, seg_maps):
        """
        Stage 4: label + center_of_mass with CuPy.

        WARNING: This stage requires host-device synchronization which can
        reduce pipeline performance. It's kept for code clarity and to match
        the documentation, despite the performance trade-off.
        """
        B = seg_maps.size(0) // self.P
        peak_positions = [[] for _ in range(B)]

        cupy_stream = cp.cuda.ExternalStream(ptr=torch.cuda.current_stream().cuda_stream)
        with cupy_stream.use():
            for idx, seg_map in enumerate(seg_maps.flatten(0,1)):
                seg_map_cp = cp.asarray(seg_map, dtype=cp.float32)
                labeled_map, num_peaks = ndimage.label(seg_map_cp, self.structure)
                peak_coords = ndimage.center_of_mass(
                    seg_map_cp,
                    cp.asarray(labeled_map, dtype=cp.float32),
                    cp.arange(1, num_peaks + 1)
                )
                if len(peak_coords) > 0:
                    b = idx // self.P
                    p = idx % self.P
                    peak_positions[b].extend(
                        ([p] + peak.tolist()) for peak in peak_coords if len(peak) > 0
                    )
        return peak_positions

    def process_batch(self, batch_cpu, batch_idx):
        """
        Launch the pipeline stages asynchronously for a single batch,
        storing results in ring buffer slot `buf_idx`.
        Returns: peak_positions extracted from the segmentation maps.
        """
        buf_idx = batch_idx % self.num_overlap

        # Create events to signal each stage's completion
        stage1_h2d_done = torch.cuda.Event(enable_timing=False)
        stage2_preprocess_done = torch.cuda.Event(enable_timing=False)
        stage3_inference_done = torch.cuda.Event(enable_timing=False)
        stage4_postprocess_done = torch.cuda.Event(enable_timing=False)

        # Stage 1: H2D
        with torch.cuda.stream(self.stage1_h2d_stream):
            nvtx.range_push("Stage 1: H2D")
            gpu_batch = self.stage1_h2d(batch_cpu)
            stage1_h2d_done.record(self.stage1_h2d_stream)
            nvtx.range_pop()

        # Stage 2: Preprocess
        with torch.cuda.stream(self.stage2_preprocess_stream):
            self.stage2_preprocess_stream.wait_event(stage1_h2d_done)
            nvtx.range_push("Stage 2: Preprocess")
            preproc_out = self.stage2_preprocess(gpu_batch)
            self.preproc_outputs[buf_idx] = preproc_out
            stage2_preprocess_done.record(self.stage2_preprocess_stream)
            nvtx.range_pop()

        # Stage 3: Inference
        with torch.cuda.stream(self.stage3_inference_stream):
            self.stage3_inference_stream.wait_event(stage2_preprocess_done)
            nvtx.range_push("Stage 3: Inference")
            inf_out = self.stage3_inference(self.preproc_outputs[buf_idx])
            self.inference_outputs[buf_idx] = inf_out
            stage3_inference_done.record(self.stage3_inference_stream)
            nvtx.range_pop()

        # Stage 4: Postprocess (CuPy coordinate extraction)
        with torch.cuda.stream(self.stage4_postprocess_stream):
            self.stage4_postprocess_stream.wait_event(stage3_inference_done)
            nvtx.range_push("Stage 4: Postprocess")
            final_out = self.stage4_postprocess(self.inference_outputs[buf_idx])
            self.postprocess_outputs[buf_idx] = final_out
            stage4_postprocess_done.record(self.stage4_postprocess_stream)
            nvtx.range_pop()

        return final_out

    def process_batches(self, batches_cpu):
        """
        Dispatch multiple batches in an overlapped manner.
        After scheduling all, we wait for each final event
        and retrieve the output from the ring buffer.
        """
        final_events = []
        for i, batch_cpu in enumerate(batches_cpu):
            # Launch the pipeline for each batch
            peak_positions = self.process_batch(batch_cpu, i)
            final_events.append((i, peak_positions))

        # Gather results
        results = []
        for i, peak_positions in final_events:
            results.append(peak_positions)

        return results

    def process_batches_streaming(self, batches_cpu):
        """
        Dispatch multiple batches in a streaming manner, yielding
        peak positions as they become available.
        """
        batch_count = 0

        for batch_cpu in batches_cpu:
            peak_positions = self.process_batch(batch_cpu, batch_count)
            yield peak_positions
            batch_count += 1
