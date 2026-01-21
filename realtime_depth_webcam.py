import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import time

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

class RealtimeDepthEstimator:
    def __init__(self, encoder='vits', checkpoint_path=None):
        """
        Initialize the real-time depth estimator
        
        Args:
            encoder: Model size ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: Path to model checkpoint
        """
        print(f"Initializing Depth Anything V2 with {encoder} encoder on {DEVICE}")
        
        # Load model
        self.model = DepthAnythingV2(**model_configs[encoder])
        if checkpoint_path is None:
            checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()
        
        # Performance tracking
        self.fps = 0
        self.frame_times = []
        
    def detect_objects_simple(self, frame, depth_map):
        """
        Simple object detection using contours on depth map
        
        Returns:
            List of (bbox, mask, depth_stats) tuples
        """
        # Normalize depth map for processing
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply bilateral filter to reduce noise while preserving edges
        depth_filtered = cv2.bilateralFilter(depth_normalized, 9, 75, 75)
        
        # Apply adaptive thresholding to segment objects
        binary = cv2.adaptiveThreshold(depth_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        min_area = 1000  # Minimum area to consider as object
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create mask for this object
            mask = np.zeros(depth_map.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Extract depth statistics for this object
            object_depths = depth_map[mask == 255]
            if len(object_depths) > 0:
                depth_stats = {
                    'mean': np.mean(object_depths),
                    'median': np.median(object_depths),
                    'min': np.min(object_depths),
                    'max': np.max(object_depths),
                    'std': np.std(object_depths)
                }
                
                objects.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'mask': mask,
                    'depth_stats': depth_stats,
                    'area': area
                })
        
        return objects
    
    def detect_objects_grid(self, frame, depth_map, grid_size=(3, 3)):
        """
        Grid-based approach: divide frame into regions and get depth for each
        
        Args:
            frame: Input RGB frame
            depth_map: Depth estimation map
            grid_size: (rows, cols) for grid division
        """
        h, w = frame.shape[:2]
        grid_h, grid_w = h // grid_size[0], w // grid_size[1]
        
        regions = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                y1 = i * grid_h
                y2 = (i + 1) * grid_h if i < grid_size[0] - 1 else h
                x1 = j * grid_w
                x2 = (j + 1) * grid_w if j < grid_size[1] - 1 else w
                
                region_depth = depth_map[y1:y2, x1:x2]
                
                depth_stats = {
                    'mean': np.mean(region_depth),
                    'median': np.median(region_depth),
                    'min': np.min(region_depth),
                    'max': np.max(region_depth),
                    'std': np.std(region_depth)
                }
                
                regions.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'depth_stats': depth_stats,
                    'position': (i, j)
                })
        
        return regions
    
    def visualize_depth_overlay(self, frame, depth_map, objects, mode='contour'):
        """
        Create visualization with depth information overlay
        
        Args:
            frame: Original RGB frame
            depth_map: Depth map from model
            objects: List of detected objects with depth info
            mode: 'contour' or 'grid'
        """
        # Create colored depth map
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        # Create side-by-side visualization
        vis_frame = frame.copy()
        
        if mode == 'contour':
            # Draw contours and depth information
            for idx, obj in enumerate(objects):
                x, y, w, h = obj['bbox']
                depth_stats = obj['depth_stats']
                
                # Draw bounding box
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw contour
                cv2.drawContours(vis_frame, [obj['contour']], -1, color, 2)
                
                # Add depth text
                text_y = y - 10 if y > 30 else y + h + 20
                text = f"Obj {idx+1}: {depth_stats['mean']:.2f}"
                cv2.putText(vis_frame, text, (x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add detailed info
                detail_text = f"Min:{depth_stats['min']:.1f} Max:{depth_stats['max']:.1f}"
                cv2.putText(vis_frame, detail_text, (x, text_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        elif mode == 'grid':
            # Draw grid and depth information
            for region in objects:
                x, y, w, h = region['bbox']
                i, j = region['position']
                depth_stats = region['depth_stats']
                
                # Draw grid cell
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                # Add depth text in center of cell
                text = f"{depth_stats['mean']:.1f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y + (h + text_size[1]) // 2
                
                # Background rectangle for text
                cv2.rectangle(vis_frame, (text_x - 5, text_y - text_size[1] - 5),
                            (text_x + text_size[0] + 5, text_y + 5),
                            (0, 0, 0), -1)
                cv2.putText(vis_frame, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add FPS counter
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine original, depth map, and annotated frames
        combined = np.hstack([vis_frame, depth_colored])
        
        return combined
    
    def process_frame(self, frame, detection_mode='contour', grid_size=(3, 3)):
        """
        Process a single frame and return depth information
        
        Args:
            frame: Input BGR frame from webcam
            detection_mode: 'contour', 'grid', or 'click'
            grid_size: Grid dimensions for grid mode
            
        Returns:
            processed_frame, objects_with_depth
        """
        start_time = time.time()
        
        # Infer depth
        depth_map = self.model.infer_image(frame)
        
        # Detect objects based on mode
        if detection_mode == 'contour':
            objects = self.detect_objects_simple(frame, depth_map)
        elif detection_mode == 'grid':
            objects = self.detect_objects_grid(frame, depth_map, grid_size)
        else:
            objects = []
        
        # Visualize
        vis_frame = self.visualize_depth_overlay(frame, depth_map, objects, mode=detection_mode)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        return vis_frame, depth_map, objects
    
    def run_webcam(self, camera_id=0, detection_mode='contour', grid_size=(3, 3)):
        """
        Run real-time depth estimation on webcam feed
        
        Args:
            camera_id: Webcam device ID
            detection_mode: 'contour' or 'grid'
            grid_size: Grid dimensions for grid mode
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Starting real-time depth estimation...")
        print(f"Detection mode: {detection_mode}")
        print(f"Press 'q' to quit")
        print(f"Press 'c' to switch to contour mode")
        print(f"Press 'g' to switch to grid mode")
        print(f"Press 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame")
                    break
                
                # Process frame
                vis_frame, depth_map, objects = self.process_frame(
                    frame, detection_mode=detection_mode, grid_size=grid_size
                )
                
                # Display
                cv2.imshow('Real-time Depth Estimation', vis_frame)
                
                # Print depth information for detected objects
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"\n--- Frame {frame_count} ---")
                    for idx, obj in enumerate(objects):
                        stats = obj['depth_stats']
                        if detection_mode == 'contour':
                            print(f"Object {idx+1}: Mean={stats['mean']:.2f}, "
                                  f"Range=[{stats['min']:.2f}, {stats['max']:.2f}], "
                                  f"Area={obj.get('area', 0):.0f}")
                        else:
                            pos = obj.get('position', (0, 0))
                            print(f"Grid [{pos[0]},{pos[1]}]: Mean={stats['mean']:.2f}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    detection_mode = 'contour'
                    print("Switched to contour detection mode")
                elif key == ord('g'):
                    detection_mode = 'grid'
                    print("Switched to grid detection mode")
                elif key == ord('s'):
                    # Save current frame and depth map
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'frame_{timestamp}.jpg', frame)
                    cv2.imwrite(f'depth_{timestamp}.jpg', 
                               cv2.normalize(depth_map, None, 0, 255, 
                                           cv2.NORM_MINMAX).astype(np.uint8))
                    print(f"Saved frame and depth map with timestamp {timestamp}")
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count} frames")
            print(f"Average FPS: {self.fps:.2f}")


def main():
    """Main function to run the real-time depth estimator"""
    
    # Configuration
    ENCODER = 'vits'  # Use 'vits' for fastest, 'vitl' for better quality
    CHECKPOINT_PATH = f'checkpoints/depth_anything_v2_{ENCODER}.pth'
    CAMERA_ID = 0
    DETECTION_MODE = 'contour'  # 'contour' or 'grid'
    GRID_SIZE = (4, 4)  # For grid mode
    
    # Initialize estimator
    estimator = RealtimeDepthEstimator(encoder=ENCODER, checkpoint_path=CHECKPOINT_PATH)
    
    # Run on webcam
    estimator.run_webcam(
        camera_id=CAMERA_ID,
        detection_mode=DETECTION_MODE,
        grid_size=GRID_SIZE
    )


if __name__ == "__main__":
    main()
