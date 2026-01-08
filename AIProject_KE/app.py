import customtkinter as ctk
import cv2
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
import sys

# Force legacy Keras for compatibility with Teachable Machine models in TF 2.16+
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras
from tf_keras.models import load_model
from tf_keras.layers import DepthwiseConv2D
import threading
import os
import sys

# Set appearance mode and default color theme
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# --- Fix for Keras Model Loading Error ---
# This handles the "Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}" error
# occurring when loading models saved with different TensorFlow versions.
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' parameter if present
        super().__init__(**kwargs)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup - 100% size as per design
        self.title("AI 辨識系統")
        self.geometry("3887x2209")
        
        # Configure grid layout
        self.grid_rowconfigure(0, weight=2)
        self.grid_rowconfigure(1, weight=8)
        self.grid_columnconfigure(0, weight=1)

        # --- Header (Blue) ---
        self.header_frame = ctk.CTkFrame(self, corner_radius=200, fg_color="#4472C4") 
        self.header_frame.grid(row=0, column=0, sticky="nsew", padx=40, pady=(40, 20))
        
        self.header_label = ctk.CTkLabel(self.header_frame, text="UI 標題", font=("Microsoft JhengHei UI", 150, "bold"), text_color="white")
        self.header_label.place(relx=0.5, rely=0.5, anchor="center")

        # --- Main Content Area ---
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=40, pady=(20, 40))
        
        self.content_frame.grid_columnconfigure(0, weight=65)
        self.content_frame.grid_columnconfigure(1, weight=35)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # --- Left: Camera Feed (Green) ---
        self.camera_frame = ctk.CTkFrame(self.content_frame, corner_radius=50, fg_color="#70AD47") 
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 40), pady=0)
        
        self.camera_placeholder_label = ctk.CTkLabel(self.camera_frame, text="畫面\nCanva(畫布)", font=("Microsoft JhengHei UI", 100), text_color="white")
        self.camera_placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.video_label = ctk.CTkLabel(self.camera_frame, text="")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # --- Right: Info and Button ---
        self.right_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        # Top Right: Info Box (Orange)
        self.info_frame = ctk.CTkFrame(self.right_frame, corner_radius=150, fg_color="#ED7D31") 
        self.info_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 40))
        
        self.class_label = ctk.CTkLabel(self.info_frame, text="類別", font=("Microsoft JhengHei UI", 80), text_color="white")
        self.class_label.place(relx=0.5, rely=0.3, anchor="center")
        
        self.confidence_label = ctk.CTkLabel(self.info_frame, text="信心", font=("Microsoft JhengHei UI", 80), text_color="white")
        self.confidence_label.place(relx=0.5, rely=0.7, anchor="center")

        # Bottom Right: Button (Grey)
        self.btn_recognize = ctk.CTkButton(self.right_frame, text="辨識(BTN)", font=("Microsoft JhengHei UI", 80), 
                                           fg_color="#A5A5A5", hover_color="#808080", corner_radius=150,
                                           height=200,
                                           state="disabled")
        self.btn_recognize.grid(row=1, column=0, sticky="nsew", pady=(40, 0))

        # --- System Init ---
        self.cap = None
        self.model = None
        self.class_names = []
        self.is_running = True
        self.current_frame = None
        
        self.after(100, self.load_model_data)
        self.after(200, self.start_camera)
        self.after(1000, self.auto_recognize_loop) # Start automatic recognition loop

    def auto_recognize_loop(self):
        if self.is_running:
            if self.model is not None and self.current_frame is not None:
                self.recognize()
            # Run recognition every 200ms for real-time response
            self.after(200, self.auto_recognize_loop)

    def load_model_data(self):
        try:
            model_path = "keras_model.h5"
            labels_path = "labels.txt"
            
            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                self.class_label.configure(text="檔案缺失")
                self.confidence_label.configure(text="Check Files")
                return

            print(f"TF Version: {tf.__version__}")
            # Load model with custom object to handle 'groups' argument mismatch
            self.model = load_model(model_path, custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D}, compile=False)
            
            with open(labels_path, "r", encoding="utf-8") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print("Model loaded successfully.")
            self.class_label.configure(text="Ready")
            self.confidence_label.configure(text="--")
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Error loading model: {error_msg}")
            
            # Write full log
            with open("error_log.txt", "w") as log:
                log.write(f"TF Version: {tf.__version__}\n")
                log.write(f"Error: {error_msg}\n")
                traceback.print_exc(file=log)
            
            # Show error in UI (truncated)
            short_error = error_msg[:20] + "..." if len(error_msg) > 20 else error_msg
            self.class_label.configure(text="載入失敗")
            self.confidence_label.configure(text=short_error)

    def start_camera(self):
        try:
            # Try index 0 first, then 1 if failed
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                 self.cap = cv2.VideoCapture(1)
            
            if not self.cap.isOpened():
                print("Cannot open camera")
                self.camera_placeholder_label.configure(text="無法開啟攝像頭")
                return
            
            self.update_camera()
        except Exception as e:
            print(f"Camera error: {e}")

    def update_camera(self):
        if self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Flip frame for mirror effect (optional, feels more natural)
                frame = cv2.flip(frame, 1)
                cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = cv_image
                
                pil_image = Image.fromarray(cv_image)
                
                # Dynamic scaling to fit the camera frame
                frame_width = self.camera_frame.winfo_width()
                frame_height = self.camera_frame.winfo_height()
                
                # If window hasn't fully rendered yet, use default reasonable size for 4K
                if frame_width < 10: frame_width = 2400
                if frame_height < 10: frame_height = 1800

                # Resize image to cover the area (cover mode) or contain
                # Here we use 'contain' logic to see the whole camera feed
                img_ratio = pil_image.width / pil_image.height
                frame_ratio = frame_width / frame_height
                
                if frame_ratio > img_ratio:
                    new_height = frame_height
                    new_width = int(new_height * img_ratio)
                else:
                    new_width = frame_width
                    new_height = int(new_width / img_ratio)

                pil_image_resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                ctk_img = ctk.CTkImage(light_image=pil_image_resized, size=(new_width, new_height))
                
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img
                self.camera_placeholder_label.place_forget()
            
            self.after(30, self.update_camera)

    def recognize(self):
        if self.current_frame is None:
            self.class_label.configure(text="無影像")
            return
        
        if self.model is None:
            self.class_label.configure(text="模型未載入")
            self.load_model_data() # Try loading again
            return

        try:
            img = Image.fromarray(self.current_frame)
            size = (224, 224)
            img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            
            img_array = np.asarray(img)
            normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
            
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            
            prediction = self.model.predict(data, verbose=0)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]
            
            # Logic to remove the leading index (e.g., "0 ClassName" -> "ClassName")
            # If there's a space, take everything after the first space.
            if ' ' in class_name:
                display_name = class_name.split(' ', 1)[1]
            else:
                display_name = class_name
            
            self.class_label.configure(text=display_name.strip())
            self.confidence_label.configure(text=f"{confidence_score:.1%}")
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            self.class_label.configure(text="辨識錯誤")


    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
