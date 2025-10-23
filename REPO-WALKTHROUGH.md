# PeakNet Pipeline Data Flow Walkthrough

This document provides a comprehensive analysis of how data flows through the peaknet-pipeline system, from psana-ray input through neural network inference to final output.

## Overview

The pipeline implements a 4-stage GPU-accelerated processing system with double buffering:
1. **Stage 1**: Host-to-Device (H2D) transfer
2. **Stage 2**: Preprocessing (Crop, InstanceNorm, dimension merging)
3. **Stage 3**: Neural network inference (PeakNet model)
4. **Stage 4**: Postprocessing (segmentation map to peak coordinates) - **ACTIVE with performance trade-off**

**⚠️ Performance Note**: Stage 4 uses CuPy operations that require host-device synchronization, which reduces pipeline throughput. This stage is kept active to maintain code clarity and match this documentation, prioritizing readability over maximum performance.

## 1. Padding in the Pipeline

### Current Status: NO Padding Used

The main inference pipeline does **NOT** use padding. Here's what happens:

**Stage 2 Preprocessing** (pipeline.py:63-75):
- `Crop(H=512, W=512, "top-left")` is applied
- The Crop class (tensor_transforms.py:59-106) only crops, no automatic padding
- If input image is smaller than 512×512, the operation would fail

**Where Padding Exists** (but not used in main pipeline):
- `Pad` class in tensor_transforms.py:21-56 supports:
  - `'center'`: Pads equally on both sides
  - `'bottom-right'`: Pads only on bottom and right edges
- `Patchify` class (tensor_transforms.py:244-287) uses `Pad` internally for tiling operations

**Key Point**: The pipeline assumes input images are at least 512×512 pixels.

## 2. Complete Data Flow: psana-ray → File Writer

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT STAGE (psana-ray)                                     │
└─────────────────────────────────────────────────────────────┘
Ray Queue (input_queue_name)
  Contains: raw detector data from psana
    ↓
┌─────────────────────────────────────────────────────────────┐
│ DATA LOADING (data.py)                                      │
└─────────────────────────────────────────────────────────────┘
QueueDataset.__next__() (data.py:26-40)
  - reader.read() returns: (rank, idx, image_data, photon_energy)
  - image_data: numpy array, shape (P, H, W) where P=num_panels
  - Converts to torch.tensor
  - Adds channel dim if needed: ensures (P, H, W) format
  - Returns: (tensor, photon_energy)
    ↓
DataLoader (run_mpi.py:192-197)
  - Batches data with batch_size (e.g., 32)
  - Shape: (B, P, H, W) where B=batch_size, P=num_panels (e.g., 4)
  - pin_memory=True for faster H2D transfer
  - num_workers for parallel data loading
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PIPELINE STAGE 1: Host to Device (pipeline.py:59-61)       │
└─────────────────────────────────────────────────────────────┘
stage1_h2d_stream (dedicated CUDA stream):
  - batch_cpu.to(device, non_blocking=True)
  - Shape: (B, P, H_orig, W_orig) now on GPU
  - Non-blocking enables overlap with other operations
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PIPELINE STAGE 2: Preprocessing (pipeline.py:63-75)        │
└─────────────────────────────────────────────────────────────┘
stage2_preprocess_stream (dedicated CUDA stream):
  Input: (B, P, H_orig, W_orig)

  Transform 1: Crop(512, 512, "top-left")
    - Crops each panel to 512×512 from top-left corner
    → (B, P, 512, 512)

  Transform 2: InstanceNorm()
    - Normalizes each image independently
    - Computes mean and variance across (C, H, W) dimensions
    - Formula: (x - mean) / sqrt(var + eps)
    → (B, P, 512, 512) [normalized]

  Transform 3: MergeBatchChannelDims()
    - Reshapes (B, P, H, W) → (B*P, 1, H, W)
    - Example: B=32, P=4 → (128, 1, 512, 512)
    - Allows processing all panels independently through the model
    → (B*P, 1, 512, 512)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PIPELINE STAGE 3: Inference (pipeline.py:77-82)            │
└─────────────────────────────────────────────────────────────┘
stage3_inference_stream (dedicated CUDA stream):
  Input: (B*P, 1, 512, 512)

  model(preproc_out) with torch.amp.autocast:
    - PeakNet forward pass (ConvNeXtV2 + BiFPN architecture)
    - Mixed precision inference (bfloat16/float16 if enabled)
    → logits: (B*P, num_classes, 512, 512)

  .softmax(dim=1):
    - Convert logits to probabilities per class
    → probabilities: (B*P, num_classes, 512, 512)

  .argmax(dim=1, keepdim=True):
    - Select most likely class per pixel
    - Binary segmentation: 0=background, 1=peak
    → segmentation map: (B*P, 1, 512, 512)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PIPELINE STAGE 4: Postprocessing (pipeline.py:82-109)      │
│ NOTE: ACTIVE - See performance warning in Overview section │
└─────────────────────────────────────────────────────────────┘
stage4_postprocess_stream (dedicated CUDA stream):
  Input: seg_maps (B*P, 1, 512, 512)

  For each of the B*P panels:
    1. Convert torch tensor to CuPy array
    2. ndimage.label(seg_map, structure):
       - Finds connected components in segmentation map
       - Returns labeled_map and num_peaks
    3. ndimage.center_of_mass():
       - Computes centroid of each labeled region
       - Returns list of (y, x) coordinates
    4. Format output:
       - Add panel_id to coordinates
       - Group by original batch index

  Output: peak_positions
    Structure: List[List[List[float]]]
    - Outer list: B batches
    - Middle list: All peaks in that batch image
    - Inner list: [panel_id, y_coord, x_coord]

    Example:
    [
      [[0, 245.3, 187.2], [1, 100.5, 230.1], [2, 312.7, 156.8]],  # Batch 0
      [[0, 89.4, 421.6], [3, 201.3, 333.9]],                      # Batch 1
      ...
    ]

  PERFORMANCE TRADE-OFF:
    - CuPy operations require host-device synchronization
    - Reduces double-buffered pipeline throughput
    - Kept active for code clarity and documentation alignment
    - Historical note: Was removed in commit 8a5e255, restored for readability
    ↓
┌─────────────────────────────────────────────────────────────┐
│ RESULT ACCUMULATION (run_mpi.py:210-257)                   │
└─────────────────────────────────────────────────────────────┘
For each batch:
  # Combine all three pieces of information
  batch_results = list(zip(
    batch.cpu().numpy(),        # Original images: (B, P, H, W) numpy arrays
    peak_positions,             # Peak coords: List[List[List[float]]]
    batch_photon_energy         # Metadata: List[float]
  ))
  accumulated_results.extend(batch_results)

  # Push to Ray queue every accumulation_steps batches
  if (batch_idx + 1) % accumulation_steps == 0:
    peak_positions_queue.put(accumulated_results)
    accumulated_results = []
    ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT STAGE                                                │
└─────────────────────────────────────────────────────────────┘
Ray Queue (output_queue_name="peak_positions_queue")
  Contains accumulated tuples of:
    - Original image data (numpy array)
    - Peak coordinates [[panel, y, x], ...]
    - Photon energy (float)

  Termination signal:
    - After all data processed, rank 0 sends TERMINATION_SIGNAL
    - Sent num_consumers times for multiple downstream workers
    ↓
File Writer (external consumer, not in this repo)
  - Reads from peak_positions_queue
  - Unpacks (image, peaks, energy) tuples
  - Writes to disk in desired format (HDF5, CXI, etc.)
  - Consumed by Cheetah or other downstream tools
```

## 3. PeakNet Neural Network Output Flow

### Detailed Transformation Path

```
Stage 3: Model Forward Pass
├─ Input: (128, 1, 512, 512) [example: B=32, P=4 → B*P=128]
│   Each image is a single-channel cropped and normalized panel
│
├─ PeakNet Model Architecture:
│   ├─ Backbone: ConvNeXtV2 (feature extraction)
│   ├─ BiFPN: Bidirectional Feature Pyramid Network (multi-scale fusion)
│   └─ SegHead: Segmentation head (pixel-wise classification)
│
├─ Output logits: (128, num_classes, 512, 512)
│   Raw unnormalized scores for each class at each pixel
│   ↓
├─ .softmax(dim=1): Convert to probabilities
│   Ensures outputs sum to 1.0 across class dimension
│   → (128, num_classes, 512, 512)
│   ↓
└─ .argmax(dim=1, keepdim=True): Get class prediction per pixel
    Selects the class with highest probability
    → (128, 1, 512, 512) <-- Binary segmentation map
    Values: 0 (background) or 1 (peak)

Stage 4: Coordinate Extraction (in commit 8a5e255)
├─ Input: (128, 1, 512, 512) segmentation map on GPU
│
├─ CRITICAL: Dimension tracking for unpacking
│   The pipeline stores original dimensions in Stage 2 (pipeline.py:65-66):
│
│   if self.B is None:
│       self.B, self.P, _, _ = batch_gpu.shape  # Capture B and P!
│
│   This enables unpacking (B*P) → (B, P) in Stage 4.
│
├─ Process each panel independently (loop over 128 panels):
│   ├─ Convert to CuPy: torch tensor → cp.ndarray
│   │
│   ├─ ndimage.label(seg_map, structure):
│   │   - structure: 3×3 connectivity kernel (cp.ones((3,3)))
│   │   - Finds connected components (blob detection)
│   │   - Returns: (labeled_map, num_peaks)
│   │   - labeled_map: each pixel labeled with region ID (0, 1, 2, ...)
│   │
│   ├─ ndimage.center_of_mass():
│   │   - Computes weighted centroid of each labeled region
│   │   - Input: original seg_map values as weights
│   │   - Returns: list of (y, x) tuples, one per peak
│   │
│   └─ Format coordinates with UNPACKING LOGIC:
│
│       THE UNPACKING (B*P) → (B, P) happens here via integer arithmetic:
│
│       # Recover original batch size
│       B = seg_maps.size(0) // self.P  # e.g., 128 // 4 = 32
│
│       # Initialize output grouped by batch
│       peak_positions = [[] for _ in range(B)]
│
│       # For each flattened panel index
│       for idx, seg_map in enumerate(seg_maps.flatten(0,1)):
│           b = idx // self.P  # Batch index
│           p = idx % self.P   # Panel index within batch
│
│           # Find peaks for this panel...
│           # Then append to correct batch with panel tag:
│           peak_positions[b].extend(
│               ([p] + peak.tolist()) for peak in peak_coords
│           )
│
│       Structure: [panel_id, y_coord, x_coord]
│
└─ Output: peak_positions
    List[List[List[float]]] grouped by original batch
    Inner coordinates are sub-pixel accurate (float values)

### Concrete Example: The Unpacking Arithmetic

Let's trace through with **B=32 batches, P=4 panels**:

After Stage 2 MergeBatchChannelDims:
- Input shape: (32, 4, 512, 512)
- Output shape: (128, 1, 512, 512)  ← Flattened B*P dimension

In Stage 4, the unpacking mapping works as:

| idx | b = idx // 4 | p = idx % 4 | Interpretation          |
|-----|--------------|-------------|-------------------------|
| 0   | 0            | 0           | Batch 0, Panel 0        |
| 1   | 0            | 1           | Batch 0, Panel 1        |
| 2   | 0            | 2           | Batch 0, Panel 2        |
| 3   | 0            | 3           | Batch 0, Panel 3        |
| 4   | 1            | 0           | Batch 1, Panel 0        |
| 5   | 1            | 1           | Batch 1, Panel 1        |
| 6   | 1            | 2           | Batch 1, Panel 2        |
| 7   | 1            | 3           | Batch 1, Panel 3        |
| ... | ...          | ...         | ...                     |
| 127 | 31           | 3           | Batch 31, Panel 3       |

**Result**: Peaks are correctly grouped back into their original batches:

```python
peak_positions = [
    # Batch 0 (all peaks from panels 0, 1, 2, 3)
    [[0, 245.3, 187.2], [1, 100.5, 230.1], [2, 312.7, 156.8], ...],
    # Batch 1 (all peaks from panels 0, 1, 2, 3)
    [[0, 89.4, 421.6], [3, 201.3, 333.9], ...],
    # ...
    # Batch 31
    [[1, 78.2, 234.5], [2, 156.9, 401.2], ...]
]
```

**Key Insight**: The stored value `self.P` from Stage 2 is essential for this arithmetic.
Without it, the unpacking from (B*P) back to (B, P) would be impossible!

Accumulation Stage:
├─ Combine three data streams:
│   1. Original images: batch.cpu().numpy()
│   2. Peak positions: from Stage 4
│   3. Photon energy: from DataLoader metadata
│
├─ Create tuples: zip(images, peaks, energies)
│
├─ Accumulate over accumulation_steps batches
│   - Reduces Ray queue operations
│   - Amortizes overhead
│
└─ Push to Ray queue → consumed by file writer
```

### Does Peaknet Output Go to Disk?

**No direct disk writes in the pipeline.**

The data flow is:
1. Segmentation maps stay on GPU (Stage 3 output)
2. Converted to peak coordinates on GPU (Stage 4, using CuPy)
3. Coordinates moved to CPU and sent to Ray queue
4. **File writer (separate process)** reads from queue and writes to disk

This design decouples inference from I/O, allowing:
- Pipeline to run at full GPU speed
- File writer to handle disk I/O independently
- Multiple consumers to read from the same queue

## 4. File Writer Information Flow

Three independent streams of information are combined before reaching the file writer:

### Stream A: Original Image Data

**Emergence**: Detector readout from psana-ray
- Source: LCLS detector (e.g., Jungfrau, ePix)
- Format: Multi-panel detector images
- Shape: (P, H, W) where P = number of panels

**Flow Through Pipeline**:
```
psana-ray detector
  → Ray queue (input_queue_name)
  → QueueDataset.reader.read() → image_data
  → torch.tensor(image_data)
  → DataLoader batching → (B, P, H, W)
  → Pipeline processing (stays in batch_cpu for later)
  → run_mpi.py:227 → batch.cpu().numpy()
```

**Preservation**:
- Kept on CPU as `batch_cpu` throughout pipeline execution
- Copied back to numpy at accumulation stage
- Sent to file writer alongside processed results

**Purpose**:
- File writer may save original images for verification
- Downstream tools may need raw data for additional processing
- Quality control and debugging

### Stream B: Peak Positions (Primary Output)

**Emergence**: Stage 4 postprocessing (CuPy operations)
- Input: Segmentation maps from neural network
- Process: Connected component analysis + centroid calculation
- Output format: `[panel_id, y_coord, x_coord]` per peak

**Flow Through Pipeline**:
```
Stage 3: Segmentation map (B*P, 1, 512, 512) on GPU
  → Stage 4: ndimage.label() → labeled regions
  → Stage 4: ndimage.center_of_mass() → peak centroids
  → Format: [panel_id, y, x] per peak
  → Group by batch: List[List[List[float]]]
  → Accumulation: zip with images and energy
  → Ray queue
```

**Data Structure**:
```python
peak_positions = [
    # Batch 0 peaks:
    [
        [0, 245.3, 187.2],  # Peak on panel 0 at (y=245.3, x=187.2)
        [1, 100.5, 230.1],  # Peak on panel 1
        [2, 312.7, 156.8],  # Peak on panel 2
    ],
    # Batch 1 peaks:
    [
        [0, 89.4, 421.6],
        [3, 201.3, 333.9],
    ],
    # ...
]
```

**Purpose**:
- **Primary output of the pipeline**
- Used by Cheetah for crystallography indexing
- Sub-pixel accuracy for precise peak localization
- Panel ID needed for multi-panel detector geometry

### Stream C: Photon Energy (Metadata)

**Emergence**: psana-ray metadata
- Source: LCLS beam energy measurement
- Captured during data acquisition
- One value per event/shot

**Flow Through Pipeline**:
```
psana-ray
  → Ray queue with metadata
  → QueueDataset: reader.read() returns (rank, idx, image_data, photon_energy)
  → DataLoader: passes through as batch_photon_energy
  → Pipeline: ignored during processing
  → Accumulation: zip with images and peak_positions
  → Ray queue
```

**Preservation**:
- Carried as metadata alongside image data
- Not used in neural network inference
- Passed directly to output without modification

**Purpose**:
- Energy-dependent corrections in downstream analysis
- X-ray wavelength calculation (λ = hc/E)
- Unit cell parameter refinement
- Essential for crystallography

### Final Aggregation

**Location**: run_mpi.py:210-257

```python
# For each batch processed
batch_results = list(zip(
    batch.cpu().numpy(),      # Stream A: (B, P, H, W) numpy arrays
    peak_positions,           # Stream B: List[List[List[float]]]
    batch_photon_energy       # Stream C: List[float]
))

# Extend accumulated results
accumulated_results.extend(batch_results)

# Push to Ray queue every accumulation_steps batches
if (batch_idx + 1) % accumulation_steps == 0:
    success = ray.get(peak_positions_queue.put.remote(accumulated_results))
    accumulated_results = []  # Reset for next accumulation
```

**Data Structure Sent to File Writer**:
```python
accumulated_results = [
    (image_0, peaks_0, energy_0),  # Event 0
    (image_1, peaks_1, energy_1),  # Event 1
    # ... accumulation_steps * batch_size events
]
```

**File Writer (External Consumer)**:
The file writer process (not in this repository):
1. Reads tuples from `peak_positions_queue`
2. Unpacks: `(image, peaks, energy)` for each event
3. Formats for output (HDF5, CXI, Cheetah stream, etc.)
4. Writes to disk with appropriate metadata
5. May integrate with downstream tools (CrystFEL, Cheetah)

## 5. Pipeline Architecture: Double Buffering

### Ring Buffer Design

The pipeline uses a ring buffer with `num_overlap` slots (default: 2-3) to enable concurrent execution of different stages on different batches:

```
Time:  t0          t1          t2          t3
       ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
Batch0 │Stage1│ →  │Stage2│ →  │Stage3│ →  │Stage4│
       └──────┘    └──────┘    └──────┘    └──────┘
                   ┌──────┐    ┌──────┐    ┌──────┐
Batch1             │Stage1│ →  │Stage2│ →  │Stage3│ → ...
                   └──────┘    └──────┘    └──────┘
                               ┌──────┐    ┌──────┐
Batch2                         │Stage1│ →  │Stage2│ → ...
                               └──────┘    └──────┘
```

Each stage runs in its own CUDA stream, allowing overlap when `num_overlap >= num_stages`.

### Stage 4 Performance Trade-off: Why It Was Temporarily Removed and Restored

**Historical Context:**

In commit 8a5e255c4f17dfbd70a8e43e5355d3f6995c82ed, Stage 4 was removed due to performance concerns:
- **Issue**: CuPy's `ndimage.label()` and `center_of_mass()` require GPU synchronization
- **Impact**: Synchronization points block the pipeline, preventing stage overlap
- **Result**: Double-buffering advantage reduced, pipeline throughput degraded

**Current Decision:**

Stage 4 has been **restored and is active** in the main branch:
- **Rationale**: Prioritizing code clarity and team understanding over maximum performance
- **Trade-off**: Accepted performance overhead for easier codebase comprehension
- **Benefit**: Code directly matches this walkthrough documentation without git archaeology
- **Alternative**: For production deployments requiring maximum throughput, consider moving coordinate extraction to CPU-based downstream processes

## 6. Current Pipeline Status (Main Branch)

### What Works:
- ✅ Data loading from psana-ray via Ray queue
- ✅ Stage 1-4: Complete pipeline from H2D transfer through coordinate extraction
- ✅ Double-buffered execution with CUDA streams (with Stage 4 sync overhead)
- ✅ Model compilation with torch.compile
- ✅ Mixed precision support (bfloat16/float16)
- ✅ Accumulation and Ray queue output to downstream file writer
- ✅ NVTX profiling annotations for performance analysis

### Performance Considerations:
- ⚠️ Stage 4 synchronization reduces throughput compared to inference-only pipeline
- ⚠️ For maximum performance, consider implementing CPU-based coordinate extraction
- ⚠️ Current implementation prioritizes clarity over peak throughput

## Key Files Reference

- `run_mpi.py:96-299` - Main inference loop and MPI coordination
- `pipeline.py:1-191` - Four-stage inference pipeline implementation with CuPy
- `data.py:7-60` - QueueDataset for psana-ray integration
- `tensor_transforms.py` (in peaknet repo) - Preprocessing transformations
- Git commit `8a5e255` - Historical Stage 4 implementation (was removed, now restored)

## Related Repositories

- `$CHEETAH_CRYSTFEL_DIR`: CrystFEL stream parser for Cheetah integration
- `$PEAKNET_DIR`: PeakNet model and transformation definitions
- psana-ray: Data acquisition and Ray queue infrastructure
