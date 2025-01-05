import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import requests
import os
from dotenv import load_dotenv
import json
import io
import numpy as np
import cv2

# Load environment variables from .env file
load_dotenv(override=True)

class FaceLandmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Landmark Detection")
        
        # Configure API credentials
        self.api_key = os.getenv('FACE_API_KEY', '').strip()
        self.api_secret = os.getenv('FACE_API_SECRET', '').strip()
        self.api_url = "https://api-us.faceplusplus.com/facepp/v3/detect"
        
        print(f"Loaded API Key: {self.api_key}")
        print(f"Loaded API Secret: {self.api_secret}")
        
        if not self.api_key or not self.api_secret:
            tk.messagebox.showerror("Error", "Missing Face++ API credentials in .env file!")
            root.destroy()
            return
        
        # Create GUI elements
        self.create_widgets()
        
        # Store the current image and landmarks
        self.current_image = None
        self.photo = None
        self.source_landmarks = None
        self.target_landmarks = None
        self.dragging = None
        self.drag_start = None
        self.selected_feature = None
        
        # Define facial feature groups
        self.feature_groups = {
            'face_contour': (0, 17),
            'left_eyebrow': (17, 22),
            'right_eyebrow': (22, 27),
            'nose_bridge': (27, 31),
            'nose_bottom': (31, 36),
            'left_eye': (36, 42),
            'right_eye': (42, 48),
            'top_lip': (48, 54),    # Split mouth into top and bottom
            'bottom_lip': (54, 60)  # Split mouth into top and bottom
        }
        
        # Define facial triangulation (indices into landmark points)
        self.triangulation = [
            # Face contour and eyebrows
            (0, 1, 17), (1, 2, 17), (2, 3, 17), (3, 4, 17),
            (4, 5, 21), (5, 6, 21), (6, 7, 21), (7, 8, 21),
            (8, 9, 22), (9, 10, 26), (10, 11, 26), (11, 12, 26),
            (12, 13, 26), (13, 14, 26), (14, 15, 26), (15, 16, 26),
            
            # Left eye
            (17, 18, 19), (19, 20, 21),
            (36, 37, 38), (36, 38, 39), (36, 39, 40), (36, 40, 41),
            
            # Right eye
            (22, 23, 24), (24, 25, 26),
            (42, 43, 44), (42, 44, 45), (42, 45, 46), (42, 46, 47),
            
            # Nose
            (27, 28, 29), (29, 30, 31), (31, 32, 33), (33, 34, 35),
            
            # Mouth outer
            (48, 49, 50), (50, 51, 52), (52, 53, 54),
            (54, 55, 56), (56, 57, 58), (48, 54, 58),
            
            # Connect nose to eyes and mouth
            (27, 31, 35), (31, 35, 54), (31, 48, 54)
        ]
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def create_widgets(self):
        # Create buttons
        self.select_btn = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_btn.pack(pady=10)
        
        # Create morph slider
        self.morph_slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, 
                                   label="Morph Amount", command=self.on_slider_change)
        self.morph_slider.pack(pady=5, padx=10, fill=tk.X)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self.root, width=1200, height=400)  # Double width for side-by-side
        self.canvas.pack(pady=10)

    def find_feature_group(self, x, y):
        if not self.target_landmarks:
            return None
            
        # Check if we clicked on any handle
        for feature, (start, end) in self.feature_groups.items():
            if end <= len(self.target_landmarks):
                points = self.target_landmarks[start:end]
                if points:
                    # Calculate center of feature
                    center_x = sum(p[0] for p in points) / len(points)
                    center_y = sum(p[1] for p in points) / len(points)
                    
                    # Check if click is within handle radius
                    handle_radius = 15
                    if ((x - center_x) ** 2 + (y - center_y) ** 2) <= (handle_radius * 2) ** 2:
                        return feature, (start, end)
        return None

    def on_click(self, event):
        # Check if click is in the target landmarks area (right side)
        if event.x > self.image_size[0]:
            # Find which feature group was clicked
            result = self.find_feature_group(event.x, event.y)
            if result:
                self.selected_feature, (start, end) = result
                self.dragging = True
                self.drag_start = (event.x, event.y)

    def on_drag(self, event):
        if self.dragging and self.selected_feature and self.target_landmarks:
            # Calculate movement
            dx = event.x - self.drag_start[0]
            dy = event.y - self.drag_start[1]
            
            # Get the range of points for the selected feature
            start, end = self.feature_groups[self.selected_feature]
            
            # Move all points in the selected feature
            for i in range(start, end):
                if i < len(self.target_landmarks):
                    x, y = self.target_landmarks[i]
                    self.target_landmarks[i] = (x + dx, y + dy)
            
            self.drag_start = (event.x, event.y)
            self.redraw()

    def on_release(self, event):
        self.dragging = None
        if self.target_landmarks:
            self.redraw()

    def on_slider_change(self, value):
        if not self.source_landmarks or not self.target_landmarks:
            return
        self.redraw()

    def morph_image(self, src_img, src_points, dst_points, alpha):
        # Convert points to numpy arrays
        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)
        
        # Calculate interpolated points
        morph_points = []
        for i in range(len(src_points)):
            x = src_points[i][0] + alpha * (dst_points[i][0] - src_points[i][0])
            y = src_points[i][1] + alpha * (dst_points[i][1] - src_points[i][1])
            morph_points.append([x, y])
        morph_points = np.float32(morph_points)
        
        # Create output image
        height, width = src_img.shape[:2]
        morphed_img = np.zeros_like(src_img)
        
        # Calculate Delaunay triangulation
        rect = (0, 0, width, height)
        subdiv = cv2.Subdiv2D(rect)
        
        # Insert points into subdiv
        for point in src_points.astype(np.int32):
            if 0 <= point[0] < width and 0 <= point[1] < height:
                subdiv.insert((int(point[0]), int(point[1])))
        
        # Get triangles
        triangles = subdiv.getTriangleList()
        
        # Process each triangle
        for triangle in triangles:
            x1, y1, x2, y2, x3, y3 = map(int, triangle)
            
            # Find indices of triangle vertices in the points list
            src_tri = np.array([(x1, y1), (x2, y2), (x3, y3)], dtype=np.float32)
            
            # Find corresponding points in destination and morphed points
            dst_tri = []
            morph_tri = []
            
            for point in src_tri:
                # Find the closest point in src_points
                distances = np.linalg.norm(src_points - point, axis=1)
                idx = np.argmin(distances)
                
                # Get corresponding points
                dst_tri.append(dst_points[idx])
                morph_tri.append(morph_points[idx])
            
            dst_tri = np.float32(dst_tri)
            morph_tri = np.float32(morph_tri)
            
            # Calculate bounding rectangle for the triangles
            rect = cv2.boundingRect(np.float32([morph_tri]))
            x, y, w, h = rect
            
            # Create masks for the triangles
            mask = np.zeros((h, w, 3), dtype=np.float32)
            morph_tri_shifted = np.float32([[p[0] - x, p[1] - y] for p in morph_tri])
            cv2.fillConvexPoly(mask, np.int32(morph_tri_shifted), (1.0, 1.0, 1.0))
            
            # Apply affine transformation
            warp_mat = cv2.getAffineTransform(src_tri, morph_tri)
            warped = cv2.warpAffine(src_img, warp_mat, (width, height))
            
            # Extract the triangle from the warped image
            warped_tri = warped[y:y+h, x:x+w]
            
            # Apply mask
            warped_tri = warped_tri * mask
            
            # Add the warped triangle to the output image
            morphed_img[y:y+h, x:x+w] = morphed_img[y:y+h, x:x+w] * (1 - mask) + warped_tri
        
        return morphed_img.astype(np.uint8)

    def draw_shapes(self, draw, points):
        if not points:
            return
            
        # Draw each feature group with different colors
        colors = {
            'face_contour': '#FF0000',  # Bright red
            'left_eyebrow': '#0000FF',  # Bright blue
            'right_eyebrow': '#0000FF', # Bright blue
            'nose_bridge': '#00FF00',   # Bright green
            'nose_bottom': '#00FF00',   # Bright green
            'left_eye': '#FF00FF',      # Bright purple
            'right_eye': '#FF00FF',     # Bright purple
            'top_lip': '#FFA500',       # Bright orange
            'bottom_lip': '#FF4500'     # OrangeRed for bottom lip
        }
        
        # Draw each feature
        for feature, (start, end) in self.feature_groups.items():
            if end <= len(points):
                feature_points = points[start:end]
                if feature_points:
                    # Draw with white outline for visibility
                    if feature in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip', 'face_contour']:
                        # Close the loop for these features
                        draw.line(feature_points + [feature_points[0]], fill='white', width=6)
                        draw.line(feature_points + [feature_points[0]], fill=colors[feature], width=4)
                    else:
                        # Open line for other features
                        draw.line(feature_points, fill='white', width=6)
                        draw.line(feature_points, fill=colors[feature], width=4)

    def draw_shapes_on_canvas(self, points):
        if not points:
            return
            
        # Colors for features
        colors = {
            'face_contour': '#FF0000',  # Bright red
            'left_eyebrow': '#0000FF',  # Bright blue
            'right_eyebrow': '#0000FF', # Bright blue
            'nose_bridge': '#00FF00',   # Bright green
            'nose_bottom': '#00FF00',   # Bright green
            'left_eye': '#FF00FF',      # Bright purple
            'right_eye': '#FF00FF',     # Bright purple
            'top_lip': '#FFA500',       # Bright orange
            'bottom_lip': '#FF4500'     # OrangeRed for bottom lip
        }
        
        # Draw each feature
        for feature, (start, end) in self.feature_groups.items():
            if end <= len(points):
                feature_points = points[start:end]
                if feature_points:
                    # Calculate center of feature for handle
                    center_x = sum(x for x, y in feature_points) / len(feature_points)
                    center_y = sum(y for x, y in feature_points) / len(feature_points)
                    
                    # Draw feature lines
                    if feature in ['left_eye', 'right_eye', 'top_lip', 'bottom_lip', 'face_contour']:
                        # Close the loop for these features
                        canvas_points = []
                        for x, y in feature_points + [feature_points[0]]:
                            canvas_points.extend([x, y])
                        self.canvas.create_line(canvas_points, fill=colors[feature], width=4)
                    else:
                        # Open line for other features
                        canvas_points = []
                        for x, y in feature_points:
                            canvas_points.extend([x, y])
                        self.canvas.create_line(canvas_points, fill=colors[feature], width=4)
                    
                    # Draw handle (large circle) at center of feature
                    handle_radius = 25  # Made handles bigger
                    self.canvas.create_oval(
                        center_x - handle_radius, center_y - handle_radius,
                        center_x + handle_radius, center_y + handle_radius,
                        fill=colors[feature], outline='white', width=3,
                        tags=(f"handle_{feature}")
                    )
                    
                    # Add feature name as text
                    self.canvas.create_text(
                        center_x, center_y,
                        text=feature.replace('_', ' ').title(),
                        fill='white',
                        font=('Arial', 10, 'bold')  # Made text bigger
                    )

    def convert_landmarks_to_points(self, landmarks):
        points = []
        # Convert dictionary landmarks to ordered list of points
        # The order matches Face++ API documentation
        landmark_order = [
            # Face contour (0-16)
            'contour_chin',
            *[f'contour_left{i}' for i in range(8, 0, -1)],  # Left side going up
            *[f'contour_right{i}' for i in range(1, 9)],     # Right side going down
            
            # Left eyebrow (17-21)
            'left_eyebrow_left_corner',
            'left_eyebrow_upper_left_quarter',
            'left_eyebrow_upper_middle',
            'left_eyebrow_upper_right_quarter',
            'left_eyebrow_right_corner',
            
            # Right eyebrow (22-26)
            'right_eyebrow_left_corner',
            'right_eyebrow_upper_left_quarter',
            'right_eyebrow_upper_middle',
            'right_eyebrow_upper_right_quarter',
            'right_eyebrow_right_corner',
            
            # Nose bridge and tip (27-30)
            'nose_left',
            'nose_middle',
            'nose_right',
            'nose_tip',
            
            # Nose bottom (31-35)
            'nose_contour_left1',
            'nose_contour_left2',
            'nose_contour_left3',
            'nose_contour_right1',
            'nose_contour_right2',
            'nose_contour_right3',
            
            # Left eye (36-41)
            'left_eye_left_corner',
            'left_eye_upper_left_quarter',
            'left_eye_top',
            'left_eye_upper_right_quarter',
            'left_eye_right_corner',
            'left_eye_lower_right_quarter',
            'left_eye_bottom',
            'left_eye_lower_left_quarter',
            
            # Right eye (42-47)
            'right_eye_left_corner',
            'right_eye_upper_left_quarter',
            'right_eye_top',
            'right_eye_upper_right_quarter',
            'right_eye_right_corner',
            'right_eye_lower_right_quarter',
            'right_eye_bottom',
            'right_eye_lower_left_quarter',
            
            # Upper lip outer (48-53)
            'mouth_left_corner',
            'mouth_upper_lip_left_contour2',
            'mouth_upper_lip_left_contour1',
            'mouth_upper_lip_top',
            'mouth_upper_lip_right_contour1',
            'mouth_upper_lip_right_contour2',
            'mouth_right_corner',
            
            # Lower lip outer (54-59)
            'mouth_right_corner',
            'mouth_lower_lip_right_contour2',
            'mouth_lower_lip_right_contour1',
            'mouth_lower_lip_bottom',
            'mouth_lower_lip_left_contour1',
            'mouth_lower_lip_left_contour2',
            'mouth_left_corner'
        ]
        
        for key in landmark_order:
            if key in landmarks:
                points.append((int(landmarks[key]['x']), int(landmarks[key]['y'])))
            else:
                print(f"Warning: Missing landmark {key}")
        
        return points

    def detect_landmarks(self, file_path):
        print(f"Using API Key: {self.api_key}")
        print(f"Using API Secret: {self.api_secret}")
        
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
            
            # Store original dimensions for scaling
            orig_width = img_copy.width
            orig_height = img_copy.height
            
            # Convert to bytes for upload
            img_buffer = io.BytesIO()
            img_copy.save(img_buffer, format='JPEG', quality=95)
            img_bytes = img_buffer.getvalue()
        
        # Prepare the API request
        files = {'image_file': ('image.jpg', img_bytes, 'image/jpeg')}
        data = {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'return_landmark': '1',
            'return_attributes': 'headpose'
        }

        try:
            # Make API request
            response = requests.post(self.api_url, files=files, data=data)
            result = response.json()
            
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")
            
            if 'error_message' in result:
                tk.messagebox.showerror("API Error", f"Face++ API Error: {result['error_message']}")
                return

            if 'faces' in result and result['faces']:
                print(f"Found {len(result['faces'])} faces")
                
                # Calculate scaling factors
                scale_x = self.image_size[0] / orig_width
                scale_y = self.image_size[1] / orig_height
                
                # Get landmarks for the first face
                face = result['faces'][0]
                if 'landmark' in face:
                    # Scale the landmarks to match our display size
                    scaled_landmarks = {}
                    for key, value in face['landmark'].items():
                        scaled_landmarks[key] = {
                            'x': int(value['x'] * scale_x),
                            'y': int(value['y'] * scale_y)
                        }
                    
                    # Convert landmarks to ordered points
                    self.source_landmarks = self.convert_landmarks_to_points(scaled_landmarks)
                    
                    # Create target landmarks with larger offset
                    offset = self.image_size[0] + 100  # Increased gap to 100px
                    self.target_landmarks = [(x + offset, y) for x, y in self.source_landmarks]
                    
                    # Draw initial state
                    self.redraw()
                else:
                    tk.messagebox.showwarning("Warning", "No landmarks found in the face data!")
            else:
                tk.messagebox.showwarning("Warning", "No faces detected in the image!")

        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            # Load and display the image
            self.load_image(file_path)
            # Detect facial landmarks
            self.detect_landmarks(file_path)

    def load_image(self, file_path):
        # Open and resize image to fit canvas
        image = Image.open(file_path)
        # Calculate resize ratio while maintaining aspect ratio
        canvas_ratio = 600/400
        image_ratio = image.width/image.height
        
        if image_ratio > canvas_ratio:
            new_width = 400  # Made base image smaller
            new_height = int(400/image_ratio)
        else:
            new_height = 400
            new_width = int(400*image_ratio)
            
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Store the current image and its size
        self.current_image = image.copy()
        self.image_size = (new_width, new_height)
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(image)
        # Make canvas wider to accommodate handles
        canvas_width = new_width * 2 + 200  # Added more space for handles
        canvas_height = new_height + 100     # Added padding for handles
        self.canvas.config(width=canvas_width, height=canvas_height)
        # Center the images vertically
        y_center = canvas_height // 2
        # Position images with gap
        self.canvas.create_image(new_width//2, y_center, image=self.photo)  # Original image
        
    def redraw(self):
        if not self.source_landmarks or not self.target_landmarks:
            return
            
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw original image with landmarks overlay
        self.canvas.create_image(self.image_size[0]//2, self.canvas.winfo_height()//2, image=self.photo)
        
        # Create a separate image for landmark visualization
        overlay_image = self.current_image.copy()
        draw = ImageDraw.Draw(overlay_image)
        self.draw_shapes(draw, self.source_landmarks)
        
        # Get current morph value
        value = self.morph_slider.get()
        
        # Convert original image (without landmarks) to numpy array for OpenCV
        img_array = np.array(self.current_image)
        if len(img_array.shape) == 3:  # Color image
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Get source and target points
        src_points = np.float32([[x, y] for x, y in self.source_landmarks])
        dst_points = np.float32([[x - self.image_size[0], y] for x, y in self.target_landmarks])
        
        # Morph the image
        morphed_img = self.morph_image(img_array, src_points, dst_points, float(value) / 100)
        
        # Convert back to RGB for PIL
        if len(morphed_img.shape) == 3:
            morphed_img = cv2.cvtColor(morphed_img, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        morphed_image = Image.fromarray(morphed_img.astype(np.uint8))
        
        # Calculate morphed landmarks for visualization
        morphed_landmarks = []
        for (sx, sy), (tx, ty) in zip(self.source_landmarks, self.target_landmarks):
            dx = tx - (sx + self.image_size[0])
            dy = ty - sy
            mx = int(sx + dx * float(value) / 100)
            my = int(sy + dy * float(value) / 100)
            morphed_landmarks.append((mx, my))
        
        # Create a separate image for morphed landmark visualization
        morphed_overlay = morphed_image.copy()
        draw = ImageDraw.Draw(morphed_overlay)
        self.draw_shapes(draw, morphed_landmarks)
        
        # Update morphed image display (clean version)
        self.morphed_photo = ImageTk.PhotoImage(morphed_image)
        self.canvas.create_image(self.image_size[0] + self.image_size[0]//2, 
                               self.canvas.winfo_height()//2, 
                               image=self.morphed_photo)
        
        # Draw target shapes and handles for interaction
        self.draw_shapes_on_canvas(self.target_landmarks)

def main():
    root = tk.Tk()
    app = FaceLandmarkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 