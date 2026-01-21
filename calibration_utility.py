"""
Camera Calibration and Metric Depth Conversion Utility

This module helps calibrate the depth estimation system with ground truth
measurements to enable metric depth measurements.
"""

import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import json
import os
from datetime import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


class DepthCalibrator:
    """Calibrate depth estimation with ground truth measurements"""
    
    def __init__(self, encoder='vits', checkpoint_path=None):
        """Initialize the calibrator"""
        print(f"Initializing Depth Calibrator with {encoder} on {DEVICE}")
        
        # Load model
        self.model = DepthAnythingV2(**model_configs[encoder])
        if checkpoint_path is None:
            checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
        
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()
        
        # Calibration data
        self.calibration_points = []
        self.calibration_params = None
        
    def capture_calibration_point(self, frame, ground_truth_distance):
        """
        Capture a calibration point
        
        Args:
            frame: Image of object at known distance
            ground_truth_distance: Actual distance in meters (or your unit)
            
        Returns:
            Estimated depth value
        """
        # Estimate depth
        depth_map = self.model.infer_image(frame)
        
        # Get center region depth (assuming object is centered)
        h, w = depth_map.shape
        center_region = depth_map[h//4:3*h//4, w//4:3*w//4]
        estimated_depth = np.median(center_region)
        
        # Store calibration point
        self.calibration_points.append({
            'estimated': estimated_depth,
            'ground_truth': ground_truth_distance,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Captured: Estimated={estimated_depth:.2f}, "
              f"Ground Truth={ground_truth_distance:.2f}m")
        
        return estimated_depth
    
    def compute_calibration(self, method='linear'):
        """
        Compute calibration parameters from collected points
        
        Args:
            method: 'linear', 'polynomial', or 'exponential'
            
        Returns:
            Dictionary with calibration parameters
        """
        if len(self.calibration_points) < 2:
            print("Need at least 2 calibration points!")
            return None
        
        # Extract data
        estimated = np.array([p['estimated'] for p in self.calibration_points])
        ground_truth = np.array([p['ground_truth'] for p in self.calibration_points])
        
        if method == 'linear':
            # Linear regression: y = mx + b
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(estimated.reshape(-1, 1), ground_truth)
            
            params = {
                'method': 'linear',
                'scale': float(model.coef_[0]),
                'offset': float(model.intercept_),
                'r2_score': float(model.score(estimated.reshape(-1, 1), ground_truth))
            }
            
        elif method == 'polynomial':
            # Polynomial fit: y = ax^2 + bx + c
            coeffs = np.polyfit(estimated, ground_truth, 2)
            
            # Calculate R² score
            predicted = np.polyval(coeffs, estimated)
            ss_res = np.sum((ground_truth - predicted) ** 2)
            ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            params = {
                'method': 'polynomial',
                'coefficients': coeffs.tolist(),
                'r2_score': float(r2)
            }
            
        elif method == 'exponential':
            # Exponential fit: y = a * exp(b*x)
            from scipy.optimize import curve_fit
            
            def exp_func(x, a, b):
                return a * np.exp(b * x)
            
            try:
                popt, _ = curve_fit(exp_func, estimated, ground_truth)
                predicted = exp_func(estimated, *popt)
                
                ss_res = np.sum((ground_truth - predicted) ** 2)
                ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                params = {
                    'method': 'exponential',
                    'a': float(popt[0]),
                    'b': float(popt[1]),
                    'r2_score': float(r2)
                }
            except:
                print("Exponential fit failed, using linear instead")
                return self.compute_calibration('linear')
        
        self.calibration_params = params
        
        # Print results
        print(f"\nCalibration Results ({method}):")
        print(f"R² Score: {params['r2_score']:.4f}")
        if method == 'linear':
            print(f"Scale: {params['scale']:.4f}")
            print(f"Offset: {params['offset']:.4f}")
            print(f"Formula: distance = {params['scale']:.4f} * depth + {params['offset']:.4f}")
        
        return params
    
    def apply_calibration(self, estimated_depth):
        """
        Convert estimated depth to calibrated distance
        
        Args:
            estimated_depth: Raw depth value from model
            
        Returns:
            Calibrated distance in real-world units
        """
        if self.calibration_params is None:
            print("Warning: No calibration parameters! Using raw depth.")
            return estimated_depth
        
        method = self.calibration_params['method']
        
        if method == 'linear':
            distance = (self.calibration_params['scale'] * estimated_depth + 
                       self.calibration_params['offset'])
        
        elif method == 'polynomial':
            coeffs = self.calibration_params['coefficients']
            distance = np.polyval(coeffs, estimated_depth)
        
        elif method == 'exponential':
            a = self.calibration_params['a']
            b = self.calibration_params['b']
            distance = a * np.exp(b * estimated_depth)
        
        return distance
    
    def save_calibration(self, filename='calibration.json'):
        """Save calibration parameters to file"""
        if self.calibration_params is None:
            print("No calibration parameters to save!")
            return
        
        data = {
            'calibration_params': self.calibration_params,
            'calibration_points': self.calibration_points,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename='calibration.json'):
        """Load calibration parameters from file"""
        if not os.path.exists(filename):
            print(f"Calibration file {filename} not found!")
            return False
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.calibration_params = data['calibration_params']
        self.calibration_points = data.get('calibration_points', [])
        
        print(f"Calibration loaded from {filename}")
        print(f"Method: {self.calibration_params['method']}")
        print(f"R² Score: {self.calibration_params['r2_score']:.4f}")
        
        return True
    
    def interactive_calibration(self, camera_id=0, num_points=5):
        """
        Interactive calibration process
        
        Args:
            camera_id: Camera device ID
            num_points: Number of calibration points to collect
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("\n" + "="*60)
        print("Interactive Calibration")
        print("="*60)
        print(f"\nInstructions:")
        print(f"1. Place an object at a known distance")
        print(f"2. Center the object in the frame")
        print(f"3. Press SPACE to capture")
        print(f"4. Enter the actual distance in meters")
        print(f"5. Repeat for {num_points} different distances")
        print(f"\nRecommended distances: 0.5m, 1.0m, 1.5m, 2.0m, 3.0m")
        print("="*60 + "\n")
        
        points_captured = 0
        
        try:
            while points_captured < num_points:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Show frame with instructions
                display = frame.copy()
                h, w = display.shape[:2]
                
                # Draw center crosshair
                cv2.line(display, (w//2 - 50, h//2), (w//2 + 50, h//2), (0, 255, 0), 2)
                cv2.line(display, (w//2, h//2 - 50), (w//2, h//2 + 50), (0, 255, 0), 2)
                
                # Draw calibration region
                cv2.rectangle(display, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
                
                # Add text
                text = f"Point {points_captured + 1}/{num_points} - Press SPACE to capture"
                cv2.putText(display, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Calibration', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Capture frame
                    print(f"\nCapturing point {points_captured + 1}...")
                    estimated = self.capture_calibration_point(frame, 0)  # Temporary
                    
                    # Get ground truth from user
                    while True:
                        try:
                            distance_str = input(f"Enter actual distance in meters: ")
                            distance = float(distance_str)
                            if distance > 0:
                                # Update the last calibration point
                                self.calibration_points[-1]['ground_truth'] = distance
                                print(f"Recorded: {distance}m")
                                points_captured += 1
                                break
                            else:
                                print("Distance must be positive!")
                        except ValueError:
                            print("Invalid input! Enter a number.")
                
                elif key == ord('q'):
                    print("Calibration cancelled")
                    break
            
            # Compute calibration
            if points_captured >= 2:
                print("\nComputing calibration...")
                self.compute_calibration('linear')
                
                # Save calibration
                self.save_calibration()
                
                # Show results
                self.visualize_calibration()
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def visualize_calibration(self):
        """Visualize calibration results"""
        if len(self.calibration_points) < 2:
            print("Not enough calibration points to visualize")
            return
        
        import matplotlib.pyplot as plt
        
        estimated = np.array([p['estimated'] for p in self.calibration_points])
        ground_truth = np.array([p['ground_truth'] for p in self.calibration_points])
        
        # Create fitted line
        x_range = np.linspace(estimated.min(), estimated.max(), 100)
        y_fitted = np.array([self.apply_calibration(x) for x in x_range])
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(estimated, ground_truth, s=100, alpha=0.6, label='Calibration Points')
        plt.plot(x_range, y_fitted, 'r-', linewidth=2, label='Fitted Curve')
        plt.xlabel('Estimated Depth (raw)', fontsize=12)
        plt.ylabel('Ground Truth Distance (m)', fontsize=12)
        plt.title(f'Depth Calibration (R² = {self.calibration_params["r2_score"]:.4f})', 
                 fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('calibration_plot.png', dpi=150)
        print("Calibration plot saved as 'calibration_plot.png'")
        plt.show()


class MetricDepthEstimator:
    """Depth estimator with calibrated metric measurements"""
    
    def __init__(self, encoder='vits', calibration_file='calibration.json'):
        """Initialize with calibration"""
        self.calibrator = DepthCalibrator(encoder=encoder)
        
        if os.path.exists(calibration_file):
            self.calibrator.load_calibration(calibration_file)
        else:
            print(f"Warning: No calibration file found at {calibration_file}")
            print("Use DepthCalibrator.interactive_calibration() first!")
    
    def measure_distance(self, frame, region=None):
        """
        Measure distance in calibrated units
        
        Args:
            frame: Input image
            region: (x, y, w, h) region to measure, or None for center
            
        Returns:
            Distance in calibrated units (meters)
        """
        # Get depth map
        depth_map = self.calibrator.model.infer_image(frame)
        
        # Extract region
        if region is None:
            h, w = depth_map.shape
            depth_region = depth_map[h//4:3*h//4, w//4:3*w//4]
        else:
            x, y, w, h = region
            depth_region = depth_map[y:y+h, x:x+w]
        
        # Calculate statistics
        estimated_depth = np.median(depth_region)
        distance = self.calibrator.apply_calibration(estimated_depth)
        
        stats = {
            'distance': distance,
            'estimated_depth': estimated_depth,
            'min': self.calibrator.apply_calibration(np.min(depth_region)),
            'max': self.calibrator.apply_calibration(np.max(depth_region)),
            'std': np.std(depth_region)
        }
        
        return stats


def main():
    """Main calibration workflow"""
    
    print("\nDepth Calibration Utility")
    print("="*60)
    print("\nOptions:")
    print("1. Perform new calibration")
    print("2. Load existing calibration")
    print("3. Test calibration")
    
    choice = input("\nEnter choice (1-3): ")
    
    calibrator = DepthCalibrator(encoder='vits')
    
    if choice == '1':
        # New calibration
        num_points = int(input("Number of calibration points (recommend 5): "))
        calibrator.interactive_calibration(camera_id=0, num_points=num_points)
    
    elif choice == '2':
        # Load existing
        filename = input("Calibration filename (default: calibration.json): ").strip()
        if not filename:
            filename = 'calibration.json'
        
        if calibrator.load_calibration(filename):
            calibrator.visualize_calibration()
    
    elif choice == '3':
        # Test calibration
        filename = input("Calibration filename (default: calibration.json): ").strip()
        if not filename:
            filename = 'calibration.json'
        
        calibrator.load_calibration(filename)
        
        # Test on webcam
        cap = cv2.VideoCapture(0)
        print("\nTesting calibration. Press 'q' to quit, SPACE to measure.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Measure center region
            depth_map = calibrator.model.infer_image(frame)
            h, w = depth_map.shape
            center_depth = np.median(depth_map[h//4:3*h//4, w//4:3*w//4])
            calibrated_dist = calibrator.apply_calibration(center_depth)
            
            # Display
            display = frame.copy()
            cv2.rectangle(display, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            text = f"Distance: {calibrated_dist:.2f}m"
            cv2.putText(display, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Calibrated Distance', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                print(f"Measured distance: {calibrated_dist:.2f}m")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
