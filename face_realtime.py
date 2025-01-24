import cv2 as cv
import numpy as np
import joblib

# Load Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Load the trained KNN model
knn = joblib.load('knn_face_recognizer.pkl')

# Load the Team_members list
Team_members = np.load('team_members.npy', allow_pickle=True).tolist()

# Function to extract face embeddings (same as during training)
def extract_face_embeddings(face_image):
    return face_image.flatten()

# Start video capture
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces_rect:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (100, 100))  # Resize to match training size

        # Extract face embeddings
        embedding = extract_face_embeddings(roi_gray)

        # Predict using KNN
        id_ = knn.predict([embedding])[0]
        confidence = np.max(knn.predict_proba([embedding]))  # Get the confidence score

        # Set a confidence threshold (e.g., 0.7)
        if confidence > 0.3:  # Adjust threshold as needed
            name = Team_members[id_]
        else:
            name = "Unknown"

        # Draw rectangle and label on the frame
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv.destroyAllWindows()