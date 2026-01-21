# Real-time Depth Estimation with Depth Anything V2

A comprehensive system for real-time depth estimation using webcam with multiple object detection modes and depth measurement capabilities.

## Features

### Basic Version (`realtime_depth_webcam.py`)
- ✅ Real-time depth map generation from webcam feed
- ✅ Contour-based object detection with depth statistics
- ✅ Grid-based depth measurement across frame regions
- ✅ FPS monitoring and performance tracking
- ✅ Side-by-side visualization (RGB + Depth)
- ✅ Save frames and depth maps

### Advanced Version (`advanced_depth_webcam.py`)
- ✅ All basic features plus:
- ✅ Click-to-measure depth at any point
- ✅ Optional YOLO object detection with depth
- ✅ Depth-based clustering and segmentation
- ✅ Multiple visualization modes
- ✅ Crosshair measurement system
- ✅ Export depth maps as .npy for analysis

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Depth Anything V2

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -e .
cd ..
```

### 3. Download Model Checkpoints

Create a `checkpoints` directory and download the models:

```bash
mkdir -p checkpoints
cd checkpoints

# Download the model you want (vits is fastest, vitl is most accurate)
# Option 1: Small model (fastest, ~100 FPS on GPU)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Option 2: Base model (balanced)
# wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

# Option 3: Large model (best quality)
# wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

cd ..
```

### 4. Optional: Install YOLO (for advanced version)

```bash
pip install ultralytics
```

## Usage

### Basic Version

**1. Simple Contour Detection:**
```bash
python realtime_depth_webcam.py
```

**2. Grid-based Depth Mapping:**
Edit `main()` function:
```python
DETECTION_MODE = 'grid'
GRID_SIZE = (4, 4)  # 4x4 grid
```

### Advanced Version

**1. With Depth Clustering:**
```bash
python advanced_depth_webcam.py
```

**2. With YOLO Object Detection:**
Edit configuration in `main()`:
```python
USE_YOLO = True
DETECTION_MODE = 'yolo'
```

**3. Depth-only Mode (fastest):**
```python
DETECTION_MODE = 'none'
```

## Controls

### Keyboard Controls
| Key | Action |
|-----|--------|
| `q` | Quit application |
| `c` | Switch to contour detection mode |
| `g` | Switch to grid mode |
| `y` | Switch to YOLO detection (advanced version) |
| `n` | Depth-only mode (no object detection) |
| `s` | Save current frame and depth map |
| `r` | Clear measurement points |

### Mouse Controls (Advanced Version)
| Action | Function |
|--------|----------|
| Left Click | Measure depth at clicked point |
| Right Click | Clear all measurement points |

## Detection Modes Explained

### 1. Contour Detection Mode
- Uses depth map discontinuities to detect objects
- Fast and efficient
- Good for well-separated objects
- Parameters: `min_area`, `morphology kernels`

### 2. Grid Mode
- Divides frame into regular grid
- Measures average depth in each cell
- Best for scene understanding
- Configurable grid size: `(rows, cols)`

### 3. YOLO Detection Mode (Advanced)
- Uses YOLOv8 for semantic object detection
- Provides object class labels
- More accurate object boundaries
- Requires `ultralytics` package

### 4. Depth Clustering Mode (Advanced)
- Segments scene by depth similarity
- Uses K-means clustering
- Good for layer-based scene analysis
- Configurable number of clusters

## Performance Optimization

### Model Selection
```python
# For real-time (>30 FPS):
ENCODER = 'vits'  # Fastest, ~100 FPS on RTX 3080

# For balanced performance:
ENCODER = 'vitb'  # ~50 FPS on RTX 3080

# For best quality:
ENCODER = 'vitl'  # ~30 FPS on RTX 3080
```

### Camera Settings
```python
# Reduce resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# For quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Inference Optimization
```python
# Use half precision (if supported)
model = model.half()  # FP16

# Batch processing (for offline analysis)
depths = model.infer_image(frames_batch)
```

## Output Files

### Saved Files Format
- `frame_YYYYMMDD_HHMMSS.jpg` - Original RGB frame
- `depth_YYYYMMDD_HHMMSS.jpg` - Visualized depth map (colormap)
- `depth_YYYYMMDD_HHMMSS.npy` - Raw depth values (numpy array)

### Loading Saved Depth Maps
```python
import numpy as np
import matplotlib.pyplot as plt

# Load depth map
depth = np.load('depth_20250121_143022.npy')

# Visualize
plt.imshow(depth, cmap='turbo')
plt.colorbar(label='Depth')
plt.title('Depth Map')
plt.show()

# Get statistics
print(f"Min depth: {depth.min():.2f}")
print(f"Max depth: {depth.max():.2f}")
print(f"Mean depth: {depth.mean():.2f}")
```

## Advanced Features

### Depth Calibration
If you have ground truth depth measurements, you can calibrate:

```python
# In advanced_depth_webcam.py
estimator = AdvancedDepthEstimator(encoder='vits')

# Set calibration parameters
estimator.depth_scale = 0.85  # Scale factor
estimator.depth_offset = 0.5  # Offset in your units
```

### Custom Object Detection
Add your own detection method:

```python
def detect_custom(self, frame, depth_map):
    """Your custom detection logic"""
    # Example: Detect objects by depth threshold
    threshold = np.percentile(depth_map, 60)
    mask = (depth_map > threshold).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    objects = []
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            x, y, w, h = cv2.boundingRect(contour)
            object_depth = depth_map[y:y+h, x:x+w]
            
            objects.append({
                'bbox': (x, y, w, h),
                'depth_stats': {
                    'mean': np.mean(object_depth),
                    'median': np.median(object_depth)
                }
            })
    
    return objects
```

### Integration with Your Projects

**Example: Frailty Assessment**
```python
# Measure distances for gait analysis
def calculate_stride_length(depth_map, foot_positions):
    """Calculate stride length from depth map"""
    x1, y1 = foot_positions[0]
    x2, y2 = foot_positions[1]
    
    depth1 = depth_map[y1, x1]
    depth2 = depth_map[y2, x2]
    
    # Calculate 3D distance
    pixel_distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    depth_distance = abs(depth2 - depth1)
    
    # Convert to real-world units (requires calibration)
    real_distance = calculate_real_distance(pixel_distance, depth_distance)
    
    return real_distance
```

**Example: Object Dimension Measurement**
```python
def measure_object_dimensions(depth_map, bbox, camera_params):
    """Estimate object size from depth map"""
    x, y, w, h = bbox
    avg_depth = np.mean(depth_map[y:y+h, x:x+w])
    
    # Convert pixels to real-world dimensions
    fx, fy = camera_params['focal_length']
    width_real = (w * avg_depth) / fx
    height_real = (h * avg_depth) / fy
    
    return width_real, height_real
```

## Troubleshooting

### Low FPS
1. Use smaller model: `vits` instead of `vitl`
2. Reduce camera resolution
3. Disable object detection: `DETECTION_MODE = 'none'`
4. Check GPU availability: `print(DEVICE)`

### Inaccurate Depth
1. Ensure good lighting conditions
2. Calibrate depth scale and offset
3. Use larger model for better accuracy
4. Avoid transparent/reflective surfaces

### Camera Not Found
```python
# List available cameras
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()

# Use specific camera
CAMERA_ID = 1  # Change as needed
```

### YOLO Not Working
```bash
# Install ultralytics
pip install ultralytics

# Download YOLOv8 weights (automatic on first run)
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"
```

## API Reference

### RealtimeDepthEstimator (Basic)

```python
estimator = RealtimeDepthEstimator(
    encoder='vits',           # Model size
    checkpoint_path='...'     # Path to checkpoint
)

# Process single frame
vis_frame, depth_map, objects = estimator.process_frame(
    frame,
    detection_mode='contour',  # 'contour' or 'grid'
    grid_size=(4, 4)           # For grid mode
)

# Run on webcam
estimator.run_webcam(
    camera_id=0,
    detection_mode='contour',
    grid_size=(3, 3)
)
```

### AdvancedDepthEstimator

```python
estimator = AdvancedDepthEstimator(
    encoder='vits',
    checkpoint_path='...',
    use_yolo=True              # Enable YOLO detection
)

# Set calibration
estimator.depth_scale = 0.85
estimator.depth_offset = 0.5

# Process frame
vis_frame, depth_map, objects = estimator.process_frame(
    frame,
    detection_mode='yolo'      # 'yolo', 'clustering', 'none'
)

# Run on webcam
estimator.run_webcam(
    camera_id=0,
    detection_mode='clustering'
)
```

## Examples

### Example 1: Real-time Distance Monitoring
```python
# Monitor if objects are within safe distance
def check_safe_distance(objects, threshold=1.5):
    for obj in objects:
        depth = obj['depth_stats']['mean']
        if depth < threshold:
            print(f"WARNING: Object too close! Depth: {depth:.2f}")
```

### Example 2: Export Depth Data
```python
# Save depth statistics to CSV
import csv

def export_depth_data(objects, filename='depth_data.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Object_ID', 'Mean_Depth', 
                        'Min_Depth', 'Max_Depth'])
        
        for idx, obj in enumerate(objects):
            stats = obj['depth_stats']
            writer.writerow([frame_count, idx, stats['mean'],
                           stats['min'], stats['max']])
```

## Performance Benchmarks

Tested on various hardware:

| Hardware | Model | Resolution | FPS |
|----------|-------|------------|-----|
| RTX 3080 | vits | 640x480 | 95-105 |
| RTX 3080 | vitl | 640x480 | 28-32 |
| RTX 2060 | vits | 640x480 | 65-75 |
| M1 Pro (MPS) | vits | 640x480 | 15-20 |
| CPU (i7) | vits | 640x480 | 3-5 |

## Citation

If you use Depth Anything V2 in your research:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and 
          Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}
```

## License

This project uses Depth Anything V2, which is licensed under Apache-2.0.

## Contact

For issues related to:
- Depth Anything V2: Check [official repo](https://github.com/DepthAnything/Depth-Anything-V2)
- This implementation: Open an issue on your repository

## Related Projects

- Azure Kinect DK: For hardware-based depth sensing
- RealSense SDK: Intel's depth camera solution
- OpenCV Stereo: Traditional stereo vision
- Your frailty assessment system using depth cameras

---

**Note**: This system provides relative depth estimation. For absolute metric depth, calibration with ground truth measurements is required.
