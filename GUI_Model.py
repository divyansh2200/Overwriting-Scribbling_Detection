import joblib
from skimage.feature import hog
from skimage import io, color, transform
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained SVM model
model_path = "svm_model.pkl"
svm_model = joblib.load(model_path)

# Define the HOG feature extraction function
def extract_hog_features(image):
    # Define HOG parameters matching the training parameters
    hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return hog_features

# Create the main GUI window
root = tk.Tk()
root.title("Scribbling or Overwriting Detection")

# Function to load an image, process it, and display the result
def process_image():
    # Ask the user to select an image
    image_path = filedialog.askopenfilename(title="Select an image")

    if image_path:
        # Load the selected image
        image = io.imread(image_path)
        image = color.rgb2gray(image)
        image = transform.resize(image, (250, 250))  # Resize the image to (500, 500)

        # Extract HOG features
        hog_features = extract_hog_features(image)

        # Predict using the pre-trained model
        prediction = svm_model.predict([hog_features])

        if prediction[0] == 1:
            result_label.config(text="Result: Altered")
        else:
            result_label.config(text="Result: Original")

# Create a button to load and process an image
process_button = tk.Button(root, text="Load and Process Image", command=process_image)
process_button.pack()

# Label to display the result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the GUI
root.mainloop()
