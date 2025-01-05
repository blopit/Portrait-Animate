import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import requests
import os
from dotenv import load_dotenv
import json
import io

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
        
        # Store the current image
        self.current_image = None
        self.photo = None

    def create_widgets(self):
        # Create buttons
        self.select_btn = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_btn.pack(pady=10)
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self.root, width=600, height=400)
        self.canvas.pack(pady=10)

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
            new_width = 600
            new_height = int(600/image_ratio)
        else:
            new_height = 400
            new_width = int(400*image_ratio)
            
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Store the current image and its size
        self.current_image = image.copy()
        self.image_size = (new_width, new_height)
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width//2, new_height//2, image=self.photo)

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
            'return_landmark': '1'  # Changed to '1' for basic landmarks
        }

        try:
            # Make API request
            response = requests.post(self.api_url, files=files, data=data)
            result = response.json()
            
            # Print response for debugging
            print("API Response:", json.dumps(result, indent=2))

            if 'error_message' in result:
                tk.messagebox.showerror("API Error", f"Face++ API Error: {result['error_message']}")
                return

            if 'faces' in result and result['faces']:
                print(f"Found {len(result['faces'])} faces")
                # Create a copy of the image for drawing
                image_with_landmarks = self.current_image.copy()
                draw = ImageDraw.Draw(image_with_landmarks)
                
                # Calculate scaling factors
                scale_x = self.image_size[0] / orig_width
                scale_y = self.image_size[1] / orig_height
                
                # Draw landmarks for each face
                for face in result['faces']:
                    if 'landmark' in face:
                        print(f"Drawing landmarks for face")
                        # Scale the landmarks to match our display size
                        scaled_landmarks = {}
                        for key, value in face['landmark'].items():
                            scaled_landmarks[key] = {
                                'x': int(value['x'] * scale_x),
                                'y': int(value['y'] * scale_y)
                            }
                        self.draw_landmarks(draw, scaled_landmarks)
                    else:
                        print("No landmarks in face data")
                        tk.messagebox.showwarning("Warning", "No landmarks found in the face data!")
                
                # Update the display with the new image
                self.photo = ImageTk.PhotoImage(image_with_landmarks)
                self.canvas.create_image(self.image_size[0]//2, self.image_size[1]//2, image=self.photo)
            else:
                tk.messagebox.showwarning("Warning", "No faces detected in the image!")

        except requests.exceptions.RequestException as e:
            tk.messagebox.showerror("Error", f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            tk.messagebox.showerror("Error", f"Invalid API response: {str(e)}")
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def draw_landmarks(self, draw, landmarks):
        print("Drawing landmarks:", json.dumps(landmarks, indent=2))
        
        # Calculate scaling factors based on original vs resized image dimensions
        orig_width = self.current_image.width
        orig_height = self.current_image.height
        
        # Draw each landmark point
        for point_name, coords in landmarks.items():
            # Scale coordinates to match our resized image
            x = int(coords['x'])
            y = int(coords['y'])
            
            print(f"Drawing point {point_name} at ({x}, {y})")
            
            # Draw a red dot with a white outline for visibility
            radius = 3
            draw.ellipse(
                [x-radius-1, y-radius-1, x+radius+1, y+radius+1],
                fill='white'
            )
            draw.ellipse(
                [x-radius, y-radius, x+radius, y+radius],
                fill='red'
            )
        
        # Draw connecting lines for better visualization
        # Draw face contour
        contour_points = []
        for i in range(1, 10):
            if f'contour_left{i}' in landmarks:
                point = landmarks[f'contour_left{i}']
                contour_points.append((int(point['x']), int(point['y'])))
        
        if 'contour_chin' in landmarks:
            point = landmarks['contour_chin']
            contour_points.append((int(point['x']), int(point['y'])))
        
        for i in range(9, 0, -1):
            if f'contour_right{i}' in landmarks:
                point = landmarks[f'contour_right{i}']
                contour_points.append((int(point['x']), int(point['y'])))
        
        if len(contour_points) > 2:
            # Draw white outline for visibility
            draw.line(contour_points, fill='white', width=3)
            # Draw red line
            draw.line(contour_points, fill='red', width=2)
        
        # Draw eyes
        for side in ['left', 'right']:
            eye_points = []
            for part in ['corner', 'upper_quarter', 'top', 'upper_quarter', 'corner']:
                point_name = f'{side}_eye_{part}'
                if point_name in landmarks:
                    point = landmarks[point_name]
                    eye_points.append((int(point['x']), int(point['y'])))
            
            if len(eye_points) > 2:
                # Draw white outline for visibility
                draw.line(eye_points, fill='white', width=3)
                # Draw red line
                draw.line(eye_points, fill='red', width=2)
        
        # Draw eyebrows
        for side in ['left', 'right']:
            brow_points = []
            for part in ['_eyebrow_left_corner', '_eyebrow_upper_left_quarter', '_eyebrow_upper_middle', 
                        '_eyebrow_upper_right_quarter', '_eyebrow_right_corner']:
                point_name = f'{side}{part}'
                if point_name in landmarks:
                    point = landmarks[point_name]
                    brow_points.append((int(point['x']), int(point['y'])))
            
            if len(brow_points) > 2:
                # Draw white outline for visibility
                draw.line(brow_points, fill='white', width=3)
                # Draw red line
                draw.line(brow_points, fill='red', width=2)
        
        # Draw mouth
        mouth_points = []
        if 'mouth_left_corner' in landmarks:
            point = landmarks['mouth_left_corner']
            mouth_points.append((int(point['x']), int(point['y'])))
        if 'mouth_upper_lip_top' in landmarks:
            point = landmarks['mouth_upper_lip_top']
            mouth_points.append((int(point['x']), int(point['y'])))
        if 'mouth_right_corner' in landmarks:
            point = landmarks['mouth_right_corner']
            mouth_points.append((int(point['x']), int(point['y'])))
        if 'mouth_lower_lip_bottom' in landmarks:
            point = landmarks['mouth_lower_lip_bottom']
            mouth_points.append((int(point['x']), int(point['y'])))
        if len(mouth_points) > 2:
            mouth_points.append(mouth_points[0])  # Close the shape
            # Draw white outline for visibility
            draw.line(mouth_points, fill='white', width=3)
            # Draw red line
            draw.line(mouth_points, fill='red', width=2)
        
        # Draw nose
        nose_points = []
        if 'nose_left' in landmarks and 'nose_right' in landmarks and 'nose_tip' in landmarks:
            for point_name in ['nose_left', 'nose_tip', 'nose_right']:
                point = landmarks[point_name]
                nose_points.append((int(point['x']), int(point['y'])))
            
            # Draw white outline for visibility
            draw.line(nose_points, fill='white', width=3)
            # Draw red line
            draw.line(nose_points, fill='red', width=2)

def main():
    root = tk.Tk()
    app = FaceLandmarkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 