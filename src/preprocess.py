import cv2
import os

def preprocess_images(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            cv2.imwrite(os.path.join(output_dir, filename), resized)

preprocess_images('data/raw', 'data/processed')
