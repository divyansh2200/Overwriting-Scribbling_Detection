import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the CSV file.
data = pd.read_csv('Image_DataSet.csv')

# Split the dataset into training, validation, and test sets.
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

#--------------------------------------------------------------------------------
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

# Preprocess and resize the images.
def preprocess_image(image_path, target_size=(250, 250)):
    image = io.imread(image_path)
    image = rgb2gray(image)  # Convert to grayscale
    image = resize(image, target_size)
    return image

# Apply preprocessing to the datasets.
train_data['Image'] = train_data['Image_Path'].apply(lambda x: preprocess_image(x))
valid_data['Image'] = valid_data['Image_Path'].apply(lambda x: preprocess_image(x))
test_data['Image'] = test_data['Image_Path'].apply(lambda x: preprocess_image(x))

#--------------------------------------------------------------------------------

from skimage.feature import hog

# Extract HOG features from images.
def extract_hog_features(image):
    # Define HOG parameters (you can adjust these as needed).
    hog_features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_features

# Apply feature extraction to the datasets.
train_data['HOG_Features'] = train_data['Image'].apply(lambda x: extract_hog_features(x))
valid_data['HOG_Features'] = valid_data['Image'].apply(lambda x: extract_hog_features(x))
test_data['HOG_Features'] = test_data['Image'].apply(lambda x: extract_hog_features(x))

#------------------------------------------------------------------------------------------

from sklearn.svm import SVC

# Initialize the SVM model (for non-linear SVM).
svm_model = SVC(kernel='rbf', C=1)

# Initialize the Linear SVM model (for linear SVM).
# svm_model = LinearSVC(C=1.0)

# Train the model on the training data.
svm_model.fit(list(train_data['HOG_Features']), train_data['Label'])

#---------------------------------------------------------------------------------------------

# import joblib
# model_filename = "svm_model.pkl"
# joblib.dump(svm_model, model_filename)

#---------------------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score, classification_report

# Evaluate the model on the validation set.
y_valid_pred = svm_model.predict(list(valid_data['HOG_Features']))

# Calculate accuracy.
valid_accuracy = accuracy_score(valid_data['Label'], y_valid_pred)

# Print classification report for precision, recall, F1-score, etc.
print("Validation Accuracy:", valid_accuracy)
print(classification_report(valid_data['Label'], y_valid_pred))

#---------------------------------------------------------------------------------------------

# Evaluate the model on the test set.
y_test_pred = svm_model.predict(list(test_data['HOG_Features']))

# Calculate accuracy.
test_accuracy = accuracy_score(test_data['Label'], y_test_pred)

# Print classification report for precision, recall, F1-score, etc.
print("Test Accuracy:", test_accuracy)
print(classification_report(test_data['Label'], y_test_pred))






