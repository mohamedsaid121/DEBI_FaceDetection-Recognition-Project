import os
import cv2 as cv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

# List of team members
Team_members = ['Rahma', 'Zeina', 'Omar', 'Mohamed']
DIR = 'I:\DEBI\competition\Team_pictures'

# Load Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Lists to store features (embeddings) and labels
features = []
labels = []

# Function to extract face embeddings (replace this with a proper embedding extractor)
def extract_face_embeddings(face_image):
    # For simplicity, we'll use the flattened grayscale image as the embedding.
    # In practice, you should use a pre-trained deep learning model (e.g., FaceNet).
    return face_image.flatten()

def create_train():
    for person in Team_members:
        path = os.path.join(DIR, person)
        label = Team_members.index(person)  # Assign integer label based on position in Team_members

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Warning: Unable to read image {img_path}. Skipping...")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv.resize(face_roi, (100, 100))  # Resize to a fixed size
                embedding = extract_face_embeddings(face_roi)
                features.append(embedding)
                labels.append(label)

# Create training data
create_train()
print('Training done ---------------')

# Convert features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Adjust the number of neighbors as needed
knn.fit(features, labels)

# Save the trained KNN model and labels
joblib.dump(knn, 'knn_face_recognizer.pkl')
np.save('team_members.npy', np.array(Team_members))

print("KNN model and labels saved successfully.")