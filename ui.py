import tkinter as tk
from tkinter import Label, Button, ttk, Text
from PIL import Image, ImageTk
import cv2
from model_loader import initialize_models
from inference_pipeline import process_frame
import time
import json

class AirPointerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AirPointer Application")

        # Initialize models
        self.models, self.device = initialize_models()

        # Create UI elements
        self.video_label = Label(root)
        self.video_label.pack()

        self.start_button = Button(root, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack()

        self.stop_button = Button(root, text="Stop Webcam", command=self.stop_webcam, state=tk.DISABLED)
        self.stop_button.pack()

        self.status_label = Label(root, text="Status: Ready", fg="green")
        self.status_label.pack()

        # Webcam variables
        self.cap = None
        self.running = False

        # Model selection
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var)
        self.model_dropdown.pack()

        # Metrics display
        self.metrics_label = Label(root, text="Metrics: AI Time: 0 ms, FPS: 0", fg="blue")
        self.metrics_label.pack()

        # Text area for logs
        self.log_area = Text(root, height=10, width=50)
        self.log_area.pack()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Status: Error opening webcam", fg="red")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Webcam started", fg="green")
        self.update_frame()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image="")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Webcam stopped", fg="green")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.config(text="Status: Error reading frame", fg="red")
            self.stop_webcam()
            return

        # Process the frame
        start_time = time.time()
        processed_frame = process_frame(frame, {self.model_var.get(): self.models[self.model_var.get()]}, self.device)
        end_time = time.time()

        # Calculate and display metrics
        ai_time = (end_time - start_time) * 1000
        fps = 1 / (end_time - start_time)
        self.metrics_label.config(text=f"Metrics: AI Time: {ai_time:.1f} ms, FPS: {fps:.1f}")

        # Convert the frame to ImageTk format
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the video label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule the next frame update
        self.root.after(10, self.update_frame)

    def load_models(self):
        with open("config/models.json", "r") as f:
            config = json.load(f)
            models = config.get("models", [])
            self.model_dropdown["values"] = [model["name"] for model in models]

    def log_message(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        with open("logs/application.log", "a") as log_file:
            log_file.write(message + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AirPointerApp(root)
    root.mainloop()