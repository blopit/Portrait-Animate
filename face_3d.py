import numpy as np
import cv2
import requests
from dotenv import load_dotenv
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import json
import io

# Load environment variables from .env file
load_dotenv(override=True)

class Face3DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Face Tracking")
        
        # Configure Face++ API
        self.api_key = os.getenv('FACE_API_KEY', '').strip()
        self.api_secret = os.getenv('FACE_API_SECRET', '').strip()
        self.api_url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        
        print(f"Loaded API Key: {self.api_key}")
        print(f"Loaded API Secret: {self.api_secret}")
        
        if not self.api_key or not self.api_secret:
            messagebox.showerror("Error", "Missing Face++ API credentials in .env file!")
            root.destroy()
            return
        
        # Initialize variables
        self.image = None
        self.photo = None
        self.face_3d = None
        self.landmarks_3d = None
        self.canvas_size = (800, 600)
        
        # Camera/projection parameters
        self.focal_length = 1000
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.canvas_size[0]/2],
            [0, self.focal_length, self.canvas_size[1]/2],
            [0, 0, 1]
        ])
        
        # Create GUI
        self.setup_gui()
        
        # Bind mouse movement
        self.canvas.bind('<Motion>', self.on_mouse_move)

    def setup_gui(self):
        # Create upload button
        self.upload_btn = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)
        
        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1])
        self.canvas.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
            
        # Load and display image
        self.load_image(file_path)
        
        # Analyze face in 3D
        self.analyze_face_3d(file_path)

    def load_image(self, file_path):
        # Load and resize image
        image = Image.open(file_path)
        image = image.resize((800, 600), Image.Resampling.LANCZOS)
        self.image = image
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

    def analyze_face_3d(self, file_path):
        try:
            # Load and resize image if needed
            with Image.open(file_path) as img:
                # Resize if image is too large (Face++ has a 2MB limit)
                img_copy = img.copy()
                while True:
                    # Save to temporary buffer to check size
                    temp_buffer = io.BytesIO()
                    img_copy.save(temp_buffer, format='JPEG', quality=95)
                    size_in_bytes = temp_buffer.tell()
                    
                    if size_in_bytes > 1024 * 1024:  # If larger than 1MB
                        # Resize to 80% of current size
                        new_size = (int(img_copy.width * 0.8), int(img_copy.height * 0.8))
                        img_copy = img_copy.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        break
                
                # Convert to bytes for upload
                img_buffer = io.BytesIO()
                img_copy.save(img_buffer, format='JPEG', quality=95)
                img_bytes = img_buffer.getvalue()
            
            # API request
            files = {'image_file': ('image.jpg', img_bytes, 'image/jpeg')}
            data = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'return_landmark': '1',
                'return_attributes': 'headpose'
            }
            
            print("Sending request to Face++ API...")
            print(f"API URL: {self.api_url}")
            print(f"API Key length: {len(self.api_key) if self.api_key else 'None'}")
            print(f"API Secret length: {len(self.api_secret) if self.api_secret else 'None'}")
            
            response = requests.post(self.api_url, files=files, data=data)
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")
            
            result = response.json()
            
            if 'faces' in result and result['faces']:
                face = result['faces'][0]
                if 'landmark' not in face:
                    messagebox.showerror("Error", "No facial landmarks found in the response")
                    return
                    
                landmarks = face['landmark']
                
                # Get headpose data with default values if not available
                headpose = face.get('attributes', {}).get('headpose', {})
                pitch = np.radians(float(headpose.get('pitch', 0)))
                yaw = np.radians(float(headpose.get('yaw', 0)))
                roll = np.radians(float(headpose.get('roll', 0)))
                
                # Store the face data
                self.face_3d = {
                    'landmarks': landmarks,
                    'pose': {
                        'pitch': pitch,
                        'yaw': yaw,
                        'roll': roll
                    }
                }
                
                # Convert 2D landmarks to pseudo-3D
                self.landmarks_3d = {}
                for key, value in landmarks.items():
                    x = float(value.get('x', 0))
                    y = float(value.get('y', 0))
                    
                    # Estimate Z based on facial feature type
                    z = 0
                    if 'eye' in key or 'brow' in key:
                        z = 30  # Eyes and brows are forward
                    elif 'nose' in key:
                        z = 50  # Nose is most forward
                    elif 'mouth' in key:
                        z = 20  # Mouth is between
                    elif 'contour' in key:
                        z = 0   # Contour is back
                        
                    # Adjust Z based on head pose
                    z += 10 * np.sin(pitch)  # Pitch affects Z
                    z += 10 * np.cos(yaw)    # Yaw affects Z
                    
                    self.landmarks_3d[key] = np.array([x, y, z])
                
                print("3D face analysis complete")
                self.draw_face()
            else:
                error_msg = result.get('error_message', 'No faces detected in the image')
                print(f"API Error: {error_msg}")
                messagebox.showerror("Error", error_msg)
            
        except Exception as e:
            print(f"Error analyzing face: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def project_point(self, point_3d, rotation_matrix):
        # Scale the points to match canvas size
        scale = min(self.canvas_size) / 1000
        point_3d = point_3d * scale
        
        # Center the points
        center = np.array([self.canvas_size[0]/2, self.canvas_size[1]/2, 0])
        point_3d = point_3d - center
        
        # Apply rotation
        rotated_point = rotation_matrix @ point_3d
        
        # Add center back
        rotated_point = rotated_point + center
        
        # Project to 2D (perspective projection)
        z = rotated_point[2] + self.focal_length
        if abs(z) < 1e-6:
            z = 1e-6
        
        x = rotated_point[0] * self.focal_length / z
        y = rotated_point[1] * self.focal_length / z
        
        # Center in canvas
        x += self.canvas_size[0]/2
        y += self.canvas_size[1]/2
        
        return np.array([x, y])

    def calculate_rotation_matrix(self, target_point):
        if not self.landmarks_3d:
            return np.eye(3)
        
        # Calculate angles based on mouse position
        center_x = self.canvas_size[0] / 2
        center_y = self.canvas_size[1] / 2
        
        dx = target_point[0] - center_x
        dy = target_point[1] - center_y
        
        # Scale the rotation angles
        max_angle = np.pi / 4  # 45 degrees
        yaw = np.clip(dx / center_x * max_angle, -max_angle, max_angle)
        pitch = np.clip(-dy / center_y * max_angle, -max_angle, max_angle)
        
        # Create rotation matrices
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        
        # Yaw rotation (around Y axis)
        Ry = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ])
        
        # Pitch rotation (around X axis)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ])
        
        # Combine rotations
        R = Ry @ Rx
        
        return R

    def draw_face(self, rotation_matrix=None):
        if not self.landmarks_3d:
            return
            
        if rotation_matrix is None:
            rotation_matrix = np.eye(3)
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw background image
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        
        # Draw features in order (back to front)
        features_order = [
            'face_contour',  # Draw face outline first
            'mouth_outer',   # Then mouth
            'nose_bridge',   # Then nose
            'left_eye',      # Then eyes
            'right_eye'
        ]
        
        for feature in features_order:
            self.draw_feature(feature, rotation_matrix)
            
        # Draw connection lines between features
        self.draw_connections(rotation_matrix)
        
    def draw_connections(self, rotation_matrix):
        # Define connections between features
        connections = [
            ('nose_bridge', 'left_eye'),
            ('nose_bridge', 'right_eye'),
            ('left_eye', 'right_eye')
        ]
        
        for start_feature, end_feature in connections:
            if start_feature in self.feature_centers and end_feature in self.feature_centers:
                start = self.project_point(self.feature_centers[start_feature], rotation_matrix)
                end = self.project_point(self.feature_centers[end_feature], rotation_matrix)
                
                self.canvas.create_line(
                    start[0], start[1],
                    end[0], end[1],
                    fill='#808080',  # Gray color
                    width=1,
                    dash=(4, 4)  # Dashed line
                )

    @property
    def feature_centers(self):
        centers = {}
        if not self.landmarks_3d:
            return centers
            
        # Calculate center points for main features
        feature_groups = {
            'nose_bridge': ['nose_tip'],
            'left_eye': ['left_eye_center'],
            'right_eye': ['right_eye_center']
        }
        
        for feature, points in feature_groups.items():
            if all(p in self.landmarks_3d for p in points):
                centers[feature] = np.mean([self.landmarks_3d[p] for p in points], axis=0)
        
        return centers

    def draw_feature(self, feature_name, rotation_matrix):
        # Define feature points for each part
        feature_points = {
            'face_contour': [
                'contour_chin', 
                *[f'contour_left{i}' for i in range(1, 10)],
                *[f'contour_right{i}' for i in range(1, 10)]
            ],
            'left_eye': [
                'left_eye_left_corner',
                'left_eye_upper_left_quarter',
                'left_eye_top',
                'left_eye_upper_right_quarter',
                'left_eye_right_corner',
                'left_eye_lower_right_quarter',
                'left_eye_bottom',
                'left_eye_lower_left_quarter'
            ],
            'right_eye': [
                'right_eye_left_corner',
                'right_eye_upper_left_quarter',
                'right_eye_top',
                'right_eye_upper_right_quarter',
                'right_eye_right_corner',
                'right_eye_lower_right_quarter',
                'right_eye_bottom',
                'right_eye_lower_left_quarter'
            ],
            'nose_bridge': [
                'nose_contour_left1',
                'nose_contour_left2',
                'nose_contour_left3',
                'nose_tip',
                'nose_contour_right3',
                'nose_contour_right2',
                'nose_contour_right1'
            ],
            'mouth_outer': [
                'mouth_left_corner',
                'mouth_upper_lip_left_contour2',
                'mouth_upper_lip_left_contour1',
                'mouth_upper_lip_top',
                'mouth_upper_lip_right_contour1',
                'mouth_upper_lip_right_contour2',
                'mouth_right_corner',
                'mouth_lower_lip_right_contour2',
                'mouth_lower_lip_right_contour1',
                'mouth_lower_lip_bottom',
                'mouth_lower_lip_left_contour1',
                'mouth_lower_lip_left_contour2'
            ]
        }
        
        if feature_name not in feature_points:
            return
            
        # Project points
        points_2d = []
        for point_name in feature_points[feature_name]:
            if point_name in self.landmarks_3d:
                point_3d = self.landmarks_3d[point_name]
                point_2d = self.project_point(point_3d, rotation_matrix)
                points_2d.append(point_2d)
        
        if len(points_2d) < 2:  # Need at least 2 points to draw
            return
            
        # Colors for features
        colors = {
            'face_contour': '#FF0000',  # Red
            'left_eye': '#00FF00',      # Green
            'right_eye': '#00FF00',     # Green
            'nose_bridge': '#0000FF',   # Blue
            'mouth_outer': '#FF00FF'    # Purple
        }
        
        # Convert points to canvas coordinates
        canvas_points = []
        for point in points_2d:
            canvas_points.extend([float(point[0]), float(point[1])])
        
        # Draw the feature
        if len(canvas_points) >= 4:  # Need at least 2 points (4 coordinates)
            if feature_name in ['left_eye', 'right_eye', 'mouth_outer', 'face_contour']:
                # Close the loop by adding the first point again
                canvas_points.extend(canvas_points[:2])
            
            try:
                self.canvas.create_line(
                    canvas_points,
                    fill=colors[feature_name],
                    width=2,
                    smooth=True
                )
            except Exception as e:
                print(f"Error drawing {feature_name}: {str(e)}")
                print(f"Points: {canvas_points}")

    def on_mouse_move(self, event):
        if not self.landmarks_3d:
            return
            
        # Calculate rotation based on mouse position
        rotation_matrix = self.calculate_rotation_matrix((event.x, event.y))
        
        # Update display
        self.draw_face(rotation_matrix)

def main():
    root = tk.Tk()
    app = Face3DApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 