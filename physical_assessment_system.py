"""
Physical Assessment Distance Measurement System
Tailored for frailty assessment and physical therapy applications
"""

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import time
import json
from datetime import datetime
from collections import deque

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


class PhysicalAssessmentSystem:
    """
    System for measuring distances relevant to physical assessments
    - Gait analysis (stride length, step width)
    - Reach tests (functional reach distance)
    - Balance assessments (sway measurements)
    - Fall risk detection (distance monitoring)
    """
    
    def __init__(self, encoder='vits', calibration_file=None):
        """Initialize the assessment system"""
        print(f"Initializing Physical Assessment System")
        print(f"Device: {DEVICE}, Model: {encoder}")
        
        # Load depth model
        self.model = DepthAnythingV2(**model_configs[encoder])
        checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()
        
        # Load calibration if available
        self.calibration_params = None
        if calibration_file:
            self.load_calibration(calibration_file)
        
        # Measurement tracking
        self.measurement_points = []
        self.measurement_history = deque(maxlen=1000)
        
        # Assessment modes
        self.current_mode = 'gait'  # 'gait', 'reach', 'balance', 'custom'
        
        # FPS tracking
        self.fps = 0
        self.frame_times = deque(maxlen=30)
    
    def load_calibration(self, filename):
        """Load calibration parameters"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.calibration_params = data['calibration_params']
            print(f"Loaded calibration: {self.calibration_params['method']}, "
                  f"R²={self.calibration_params['r2_score']:.4f}")
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")
    
    def apply_calibration(self, depth_value):
        """Convert depth to calibrated distance"""
        if self.calibration_params is None:
            return depth_value
        
        method = self.calibration_params['method']
        
        if method == 'linear':
            return (self.calibration_params['scale'] * depth_value + 
                   self.calibration_params['offset'])
        elif method == 'polynomial':
            return np.polyval(self.calibration_params['coefficients'], depth_value)
        
        return depth_value
    
    def detect_body_keypoints(self, frame, depth_map):
        """
        Simple body keypoint detection using depth segmentation
        More sophisticated approach would use pose estimation
        """
        h, w = depth_map.shape
        
        # Segment person (assuming they're the closest object in center region)
        center_region = depth_map[h//4:3*h//4, w//4:3*w//4]
        person_depth = np.percentile(center_region, 75)  # Top 25% closest
        
        # Create person mask
        threshold_low = person_depth * 0.95
        threshold_high = person_depth * 1.05
        person_mask = ((depth_map > threshold_low) & 
                      (depth_map < threshold_high)).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (person)
        person_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(person_contour) < 5000:
            return None
        
        # Estimate key points
        M = cv2.moments(person_contour)
        if M['m00'] == 0:
            return None
        
        # Center of mass
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Bounding box
        x, y, bw, bh = cv2.boundingRect(person_contour)
        
        # Estimate keypoints (simple heuristics)
        keypoints = {
            'center': (cx, cy),
            'top': (cx, y),  # Head approximate
            'bottom': (cx, y + bh),  # Feet approximate
            'left': (x, cy),
            'right': (x + bw, cy),
            'bbox': (x, y, bw, bh)
        }
        
        return keypoints
    
    def measure_gait_parameters(self, depth_map, keypoints):
        """
        Measure gait-related parameters
        - Subject distance from camera
        - Estimated stride length (if multiple steps detected)
        """
        if keypoints is None:
            return None
        
        # Get depth at feet position
        bottom_y = keypoints['bottom'][1]
        bottom_x = keypoints['bottom'][0]
        
        # Sample area around feet
        region_size = 20
        y1 = max(0, bottom_y - region_size)
        y2 = min(depth_map.shape[0], bottom_y + region_size)
        x1 = max(0, bottom_x - region_size)
        x2 = min(depth_map.shape[1], bottom_x + region_size)
        
        feet_depth = np.median(depth_map[y1:y2, x1:x2])
        distance = self.apply_calibration(feet_depth)
        
        # Get body height estimate
        body_height_pixels = keypoints['bbox'][3]
        
        gait_params = {
            'distance': distance,
            'body_height_pixels': body_height_pixels,
            'timestamp': time.time()
        }
        
        return gait_params
    
    def measure_reach_distance(self, depth_map, keypoints):
        """
        Measure functional reach distance
        Useful for balance and fall risk assessment
        """
        if keypoints is None:
            return None
        
        # Get arm reach (right side of bbox as proxy)
        right_x = keypoints['right'][0]
        center_y = keypoints['center'][1]
        
        # Sample depth at reach point
        region_size = 20
        y1 = max(0, center_y - region_size)
        y2 = min(depth_map.shape[0], center_y + region_size)
        x1 = max(0, right_x - region_size)
        x2 = min(depth_map.shape[1], right_x + region_size)
        
        reach_depth = np.median(depth_map[y1:y2, x1:x2])
        
        # Compare to body center depth
        center_depth = depth_map[keypoints['center'][1], keypoints['center'][0]]
        
        # Reach distance is the difference
        reach_distance = abs(self.apply_calibration(center_depth) - 
                            self.apply_calibration(reach_depth))
        
        reach_params = {
            'reach_distance': reach_distance,
            'reach_depth': self.apply_calibration(reach_depth),
            'center_depth': self.apply_calibration(center_depth),
            'timestamp': time.time()
        }
        
        return reach_params
    
    def measure_balance_sway(self, depth_history, window_size=30):
        """
        Measure body sway from depth history
        Larger sway may indicate balance issues
        """
        if len(depth_history) < window_size:
            return None
        
        recent_depths = list(depth_history)[-window_size:]
        depths_array = np.array([d['center_depth'] for d in recent_depths 
                                if 'center_depth' in d])
        
        if len(depths_array) < window_size:
            return None
        
        sway_params = {
            'mean': np.mean(depths_array),
            'std': np.std(depths_array),
            'range': np.max(depths_array) - np.min(depths_array),
            'max_deviation': np.max(np.abs(depths_array - np.mean(depths_array)))
        }
        
        return sway_params
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for manual measurements"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if hasattr(self, 'current_depth_map'):
                depth_map = self.current_depth_map
                h, w = depth_map.shape
                
                if 0 <= x < w and 0 <= y < h:
                    # Get depth with neighborhood
                    size = 10
                    y1 = max(0, y - size)
                    y2 = min(h, y + size + 1)
                    x1 = max(0, x - size)
                    x2 = min(w, x + size + 1)
                    
                    depth = np.median(depth_map[y1:y2, x1:x2])
                    distance = self.apply_calibration(depth)
                    
                    self.measurement_points.append({
                        'position': (x, y),
                        'depth': depth,
                        'distance': distance,
                        'timestamp': time.time()
                    })
                    
                    print(f"Point at ({x}, {y}): {distance:.3f}m")
                    
                    # Keep last 10 points
                    if len(self.measurement_points) > 10:
                        self.measurement_points.pop(0)
    
    def visualize_assessment(self, frame, depth_map, keypoints, measurements):
        """Create comprehensive visualization"""
        h, w = frame.shape[:2]
        vis_frame = frame.copy()
        
        # Create depth visualization
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, 
                                        cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        
        # Draw keypoints if available
        if keypoints:
            # Draw bbox
            x, y, bw, bh = keypoints['bbox']
            cv2.rectangle(vis_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            
            # Draw key points
            for name, point in keypoints.items():
                if name != 'bbox':
                    cv2.circle(vis_frame, point, 5, (0, 255, 255), -1)
        
        # Draw measurement points
        for i, point in enumerate(self.measurement_points):
            pos = point['position']
            dist = point['distance']
            
            cv2.drawMarker(vis_frame, pos, (255, 0, 255), 
                          cv2.MARKER_CROSS, 20, 2)
            
            text = f"P{i+1}: {dist:.2f}m"
            cv2.putText(vis_frame, text, (pos[0] + 10, pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Display measurements based on mode
        info_y = 30
        cv2.putText(vis_frame, f"Mode: {self.current_mode.upper()}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info_y += 30
        
        if measurements:
            if self.current_mode == 'gait' and 'distance' in measurements:
                text = f"Distance: {measurements['distance']:.2f}m"
                cv2.putText(vis_frame, text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                info_y += 25
                
                text = f"Height (px): {measurements['body_height_pixels']}"
                cv2.putText(vis_frame, text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            elif self.current_mode == 'reach' and 'reach_distance' in measurements:
                text = f"Reach: {measurements['reach_distance']:.3f}m"
                cv2.putText(vis_frame, text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            elif self.current_mode == 'balance' and measurements:
                text = f"Sway: {measurements.get('std', 0):.4f}m"
                cv2.putText(vis_frame, text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                info_y += 25
                
                text = f"Range: {measurements.get('range', 0):.4f}m"
                cv2.putText(vis_frame, text, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        cv2.putText(vis_frame, f"FPS: {self.fps:.1f}", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Combine views
        combined = np.hstack([vis_frame, depth_colored])
        
        return combined
    
    def process_frame(self, frame):
        """Process single frame"""
        start_time = time.time()
        
        # Get depth map
        depth_map = self.model.infer_image(frame)
        self.current_depth_map = depth_map
        
        # Detect keypoints
        keypoints = self.detect_body_keypoints(frame, depth_map)
        
        # Measure based on current mode
        measurements = None
        if keypoints:
            if self.current_mode == 'gait':
                measurements = self.measure_gait_parameters(depth_map, keypoints)
                if measurements:
                    self.measurement_history.append(measurements)
            
            elif self.current_mode == 'reach':
                measurements = self.measure_reach_distance(depth_map, keypoints)
                if measurements:
                    self.measurement_history.append(measurements)
            
            elif self.current_mode == 'balance':
                # Store center depth for sway calculation
                if 'center' in keypoints:
                    cx, cy = keypoints['center']
                    center_depth = self.apply_calibration(depth_map[cy, cx])
                    self.measurement_history.append({'center_depth': center_depth})
                
                # Calculate sway
                measurements = self.measure_balance_sway(self.measurement_history)
        
        # Visualize
        vis_frame = self.visualize_assessment(frame, depth_map, keypoints, measurements)
        
        # Update FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        
        return vis_frame, measurements
    
    def run_assessment(self, camera_id=0):
        """Run real-time assessment"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("Physical Assessment System")
        print("="*60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  'g' - Gait assessment mode")
        print("  'r' - Reach assessment mode")
        print("  'b' - Balance assessment mode")
        print("  's' - Save current measurements")
        print("  'c' - Clear measurement points")
        print("  Click - Manual distance measurement")
        print("="*60 + "\n")
        
        window_name = 'Physical Assessment'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                vis_frame, measurements = self.process_frame(frame)
                cv2.imshow(window_name, vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    self.current_mode = 'gait'
                    self.measurement_history.clear()
                    print("Switched to GAIT assessment mode")
                elif key == ord('r'):
                    self.current_mode = 'reach'
                    self.measurement_history.clear()
                    print("Switched to REACH assessment mode")
                elif key == ord('b'):
                    self.current_mode = 'balance'
                    self.measurement_history.clear()
                    print("Switched to BALANCE assessment mode")
                elif key == ord('c'):
                    self.measurement_points.clear()
                    print("Cleared measurement points")
                elif key == ord('s'):
                    self.save_measurements()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def save_measurements(self):
        """Save measurements to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"assessment_{self.current_mode}_{timestamp}.json"
        
        data = {
            'mode': self.current_mode,
            'timestamp': timestamp,
            'measurement_points': self.measurement_points,
            'measurement_history': list(self.measurement_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved measurements to {filename}")


def main():
    """Main function"""
    
    # Configuration
    ENCODER = 'vits'
    CALIBRATION_FILE = 'calibration.json'
    CAMERA_ID = 0
    
    # Initialize system
    system = PhysicalAssessmentSystem(
        encoder=ENCODER,
        calibration_file=CALIBRATION_FILE if os.path.exists(CALIBRATION_FILE) else None
    )
    
    # Run assessment
    system.run_assessment(camera_id=CAMERA_ID)


if __name__ == "__main__":
    import os
    main()
