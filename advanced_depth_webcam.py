import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import time
from collections import deque

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


class AdvancedDepthEstimator:
    def __init__(self, encoder='vits', checkpoint_path=None, use_yolo=False):
        """
        Advanced real-time depth estimator with multiple detection modes
        
        Args:
            encoder: Model size ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: Path to model checkpoint
            use_yolo: Whether to use YOLO for object detection
        """
        print(f"Initializing Depth Anything V2 with {encoder} encoder on {DEVICE}")
        
        # Load depth model
        self.model = DepthAnythingV2(**model_configs[encoder])
        if checkpoint_path is None:
            checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()
        
        # Load YOLO if requested
        self.use_yolo = use_yolo
        self.yolo_model = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8n.pt')  # Use nano model for speed
                print("YOLO model loaded successfully")
            except ImportError:
                print("Warning: ultralytics not installed. Install with: pip install ultralytics")
                self.use_yolo = False
        
        # Performance tracking
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
        # Click-to-measure variables
        self.click_points = []
        self.current_depth_map = None
        self.current_frame = None
        
        # Depth calibration (if you have calibration data)
        self.depth_scale = 1.0  # Adjust based on your calibration
        self.depth_offset = 0.0
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for click-to-measure"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_depth_map is not None:
                h, w = self.current_depth_map.shape
                if 0 <= x < w and 0 <= y < h:
                    # Get depth at clicked point (with small neighborhood average)
                    neighborhood_size = 5
                    y1 = max(0, y - neighborhood_size)
                    y2 = min(h, y + neighborhood_size + 1)
                    x1 = max(0, x - neighborhood_size)
                    x2 = min(w, x + neighborhood_size + 1)
                    
                    depth_region = self.current_depth_map[y1:y2, x1:x2]
                    depth_value = np.mean(depth_region)
                    
                    # Apply calibration
                    calibrated_depth = (depth_value * self.depth_scale) + self.depth_offset
                    
                    self.click_points.append({
                        'pos': (x, y),
                        'depth': depth_value,
                        'calibrated_depth': calibrated_depth
                    })
                    
                    print(f"Point ({x}, {y}): Depth = {depth_value:.2f}, "
                          f"Calibrated = {calibrated_depth:.2f}")
                    
                    # Keep only last 10 points
                    if len(self.click_points) > 10:
                        self.click_points.pop(0)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to clear points
            self.click_points.clear()
            print("Cleared all measurement points")
    
    def detect_with_yolo(self, frame, depth_map):
        """
        Detect objects using YOLO and extract depth information
        
        Returns:
            List of detected objects with depth stats
        """
        if not self.use_yolo or self.yolo_model is None:
            return []
        
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]
                
                # Extract depth for this object
                object_depth = depth_map[y1:y2, x1:x2]
                
                if object_depth.size > 0:
                    depth_stats = {
                        'mean': np.mean(object_depth),
                        'median': np.median(object_depth),
                        'min': np.min(object_depth),
                        'max': np.max(object_depth),
                        'std': np.std(object_depth)
                    }
                    
                    objects.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'confidence': float(conf),
                        'class_name': class_name,
                        'class_id': cls,
                        'depth_stats': depth_stats
                    })
        
        return objects
    
    def detect_with_depth_clustering(self, frame, depth_map, num_clusters=5):
        """
        Segment scene into depth-based clusters
        
        Args:
            frame: Input frame
            depth_map: Depth estimation
            num_clusters: Number of depth clusters
            
        Returns:
            List of depth clusters with statistics
        """
        from sklearn.cluster import KMeans
        
        # Reshape depth map for clustering
        h, w = depth_map.shape
        depth_flat = depth_map.reshape(-1, 1)
        
        # Perform K-means clustering on depth values
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(depth_flat)
        labels = labels.reshape(h, w)
        
        clusters = []
        for i in range(num_clusters):
            mask = (labels == i).astype(np.uint8) * 255
            
            # Find contours for this cluster
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour for this cluster
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 1000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    cluster_depths = depth_map[labels == i]
                    depth_stats = {
                        'mean': np.mean(cluster_depths),
                        'median': np.median(cluster_depths),
                        'min': np.min(cluster_depths),
                        'max': np.max(cluster_depths),
                        'std': np.std(cluster_depths)
                    }
                    
                    clusters.append({
                        'cluster_id': i,
                        'bbox': (x, y, w, h),
                        'contour': largest_contour,
                        'mask': mask,
                        'depth_stats': depth_stats,
                        'area': area,
                        'center_depth': kmeans.cluster_centers_[i][0]
                    })
        
        # Sort by depth (closest first)
        clusters.sort(key=lambda x: x['center_depth'], reverse=True)
        
        return clusters
    
    def visualize_advanced(self, frame, depth_map, objects, mode='yolo'):
        """
        Advanced visualization with multiple display options
        """
        # Create colored depth map
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # Draw detected objects
        for idx, obj in enumerate(objects):
            if mode == 'yolo':
                x, y, w, h = obj['bbox']
                conf = obj.get('confidence', 1.0)
                class_name = obj.get('class_name', f'Object {idx+1}')
                depth_stats = obj['depth_stats']
                
                # Color based on depth (closer = red, farther = blue)
                depth_ratio = (depth_stats['mean'] - depth_map.min()) / (depth_map.max() - depth_map.min())
                color = (int(255 * (1 - depth_ratio)), int(128 * depth_ratio), int(255 * depth_ratio))
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with depth
                label = f"{class_name} ({conf:.2f})"
                depth_label = f"D: {depth_stats['mean']:.1f}"
                
                # Background for text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(vis_frame, (x, y - 40), (x + text_size[0] + 10, y), color, -1)
                
                cv2.putText(vis_frame, label, (x + 5, y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis_frame, depth_label, (x + 5, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            elif mode == 'clustering':
                x, y, w, h = obj['bbox']
                cluster_id = obj.get('cluster_id', idx)
                depth_stats = obj['depth_stats']
                
                # Color based on cluster
                color = tuple(np.random.RandomState(cluster_id * 50).randint(0, 255, 3).tolist())
                
                # Draw contour
                cv2.drawContours(vis_frame, [obj['contour']], -1, color, 2)
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 1)
                
                # Add depth label
                label = f"Cluster {cluster_id+1}"
                depth_label = f"D: {depth_stats['mean']:.1f}"
                cv2.putText(vis_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis_frame, depth_label, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw click points
        for point in self.click_points:
            pos = point['pos']
            depth = point['calibrated_depth']
            
            # Draw crosshair
            cv2.drawMarker(vis_frame, pos, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            
            # Draw depth value
            text = f"{depth:.2f}"
            cv2.putText(vis_frame, text, (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add FPS and info
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Objects: {len(objects)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, "Click to measure depth", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Create side-by-side view
        combined = np.hstack([vis_frame, depth_colored])
        
        return combined
    
    def process_frame(self, frame, detection_mode='yolo'):
        """
        Process frame with selected detection mode
        
        Args:
            frame: Input frame
            detection_mode: 'yolo', 'clustering', or 'none'
            
        Returns:
            Visualization frame, depth map, detected objects
        """
        start_time = time.time()
        
        # Store for click callback
        self.current_frame = frame.copy()
        
        # Infer depth
        depth_map = self.model.infer_image(frame)
        self.current_depth_map = depth_map
        
        # Detect objects based on mode
        if detection_mode == 'yolo' and self.use_yolo:
            objects = self.detect_with_yolo(frame, depth_map)
        elif detection_mode == 'clustering':
            objects = self.detect_with_depth_clustering(frame, depth_map)
        else:
            objects = []
        
        # Visualize
        vis_frame = self.visualize_advanced(frame, depth_map, objects, mode=detection_mode)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        return vis_frame, depth_map, objects
    
    def run_webcam(self, camera_id=0, detection_mode='yolo'):
        """
        Run real-time depth estimation on webcam
        
        Args:
            camera_id: Camera device ID
            detection_mode: 'yolo', 'clustering', or 'none'
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"\n{'='*60}")
        print(f"Real-time Depth Estimation System")
        print(f"{'='*60}")
        print(f"Detection mode: {detection_mode}")
        print(f"Device: {DEVICE}")
        print(f"\nControls:")
        print(f"  'q' - Quit")
        print(f"  'y' - YOLO detection mode")
        print(f"  'c' - Depth clustering mode")
        print(f"  'n' - No detection (depth only)")
        print(f"  's' - Save frame + depth map")
        print(f"  'r' - Clear measurement points")
        print(f"  Left click - Measure depth at point")
        print(f"  Right click - Clear all points")
        print(f"{'='*60}\n")
        
        # Setup mouse callback
        window_name = 'Advanced Depth Estimation'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame")
                    break
                
                # Process frame
                vis_frame, depth_map, objects = self.process_frame(frame, detection_mode)
                
                # Display
                cv2.imshow(window_name, vis_frame)
                
                # Print object information periodically
                if frame_count % 60 == 0 and objects:
                    print(f"\n--- Frame {frame_count} ---")
                    for idx, obj in enumerate(objects[:5]):  # Show top 5
                        stats = obj['depth_stats']
                        if detection_mode == 'yolo':
                            print(f"{obj['class_name']}: "
                                  f"Depth={stats['mean']:.2f}±{stats['std']:.2f}, "
                                  f"Conf={obj['confidence']:.2f}")
                        else:
                            print(f"Object {idx+1}: "
                                  f"Depth={stats['mean']:.2f}±{stats['std']:.2f}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('y') and self.use_yolo:
                    detection_mode = 'yolo'
                    print("Switched to YOLO detection mode")
                elif key == ord('c'):
                    detection_mode = 'clustering'
                    print("Switched to depth clustering mode")
                elif key == ord('n'):
                    detection_mode = 'none'
                    print("Switched to depth-only mode")
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'frame_{timestamp}.jpg', frame)
                    depth_img = cv2.normalize(depth_map, None, 0, 255, 
                                             cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(f'depth_{timestamp}.jpg', depth_img)
                    np.save(f'depth_{timestamp}.npy', depth_map)
                    print(f"Saved frame and depth map: {timestamp}")
                elif key == ord('r'):
                    self.click_points.clear()
                    print("Cleared measurement points")
                
                frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nProcessed {frame_count} frames")
            print(f"Average FPS: {self.fps:.2f}")


def main():
    """Main function"""
    
    # Configuration
    ENCODER = 'vits'  # Use 'vits' for speed, 'vitl' for quality
    CHECKPOINT_PATH = f'checkpoints/depth_anything_v2_{ENCODER}.pth'
    CAMERA_ID = 0
    USE_YOLO = False  # Set to True if you have ultralytics installed
    DETECTION_MODE = 'clustering'  # 'yolo', 'clustering', or 'none'
    
    # Initialize
    estimator = AdvancedDepthEstimator(
        encoder=ENCODER,
        checkpoint_path=CHECKPOINT_PATH,
        use_yolo=USE_YOLO
    )
    
    # Run
    estimator.run_webcam(camera_id=CAMERA_ID, detection_mode=DETECTION_MODE)


if __name__ == "__main__":
    main()
