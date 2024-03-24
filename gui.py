import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import uuid
import pandas as pd
import ocr

class LicensePlateDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Detector")

        # Variables
        self.cap = None
        self.harcascade = "model\haarcascade_russian_plate_number.xml"
        self.min_area = 500
        self.data = {'UUID': [], 'Image_Path': [], 'Plate_Text': []}

        # UI Components
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.open_camera_button = ttk.Button(self.main_frame, text="Open Camera", command=self.open_camera)
        self.open_camera_button.grid(row=0, column=0, pady=5)
        
        self.perform_ocr_button = ttk.Button(self.main_frame, text="Perform OCR", command=self.perform_ocr)
        self.perform_ocr_button.grid(row=0, column=1, pady=5)

        # Bindings
        self.root.bind('s', self.save_plate_image)
        self.root.bind('<Escape>', self.quit_app)

    def open_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.show_frame()

    def show_frame(self):
        _, frame = self.cap.read()
        frame = cv2.resize(frame, (640, 480))
        self.frame = frame
        self.detect_and_highlight_plates()
        self.display_image()
        self.root.after(10, self.show_frame)

    def display_image(self):
        img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        if hasattr(self, 'panel'):
            self.panel.destroy()
        self.panel = tk.Label(self.main_frame, image=img_tk)
        self.panel.image = img_tk
        self.panel.grid(row=1, columnspan=2)

    def detect_and_highlight_plates(self):
        plates = self.detect_plates()
        for (x, y, w, h) in plates:
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def save_plate_image(self, event):
        if self.cap is not None and self.cap.isOpened():
            plates = self.detect_plates()
            print("Detected Plates:", plates)
            for (x, y, w, h) in plates:
                area = w * h
                if area > self.min_area:
                    img_roi = self.frame[y: y + h, x:x + w]
                    uid, image_path = self.save_image(img_roi)
                    plate_text = ocr.perform_ocr(image_path)
                    print("Saved Plate:", plate_text)
                    self.data['UUID'].append(uid)
                    self.data['Image_Path'].append(image_path)
                    self.data['Plate_Text'].append(plate_text)

    def detect_plates(self):
        plate_cascade = cv2.CascadeClassifier(self.harcascade)
        img_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
        return plates

    def save_image(self, img_roi):
        uid = str(uuid.uuid4())
        image_path = "plates/scanned_img_" + uid + ".jpg"
        cv2.imwrite(image_path, img_roi)
        return uid, image_path

    def perform_ocr(self):
        if 'Image_Path' in self.data and self.data['Image_Path']:
            last_image_path = self.data['Image_Path'][-1]
            plate_text = ocr.perform_ocr(last_image_path)
            result_window = tk.Toplevel(self.root)
            result_window.title("OCR Result")
            result_label = tk.Label(result_window, text=f"Extracted Plate: {plate_text}")
            result_label.pack(padx=10, pady=10)
            result_image = Image.open(last_image_path)
            result_image.thumbnail((400, 400))  # Resize image if needed
            img_tk = ImageTk.PhotoImage(result_image)
            image_label = tk.Label(result_window, image=img_tk)
            image_label.image = img_tk
            image_label.pack(padx=10, pady=10)
            
            # Print data for debugging
            print("Data:", self.data)
            
            # Save data to Excel file
            df = pd.DataFrame(self.data)
            excel_file = 'number_plates_data.xlsx'
            df.to_excel(excel_file, index=False)
            print("Data saved to Excel file:", excel_file)
        else:
            messagebox.showwarning("No Image", "No plate image available to perform OCR.")

    def quit_app(self, event=None):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = LicensePlateDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
