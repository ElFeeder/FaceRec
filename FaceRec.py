import face_recognition
import os
import cv2


DIR_KNOWN = "known"     # Directories where the faces are
DIR_UNKNOWN = "unknown"

tolerance = 0.6         # Tolerance of a match (% of accuracy permitted)

frameThickness = 3      # These two are to draw rectangles around the faces
fontThickness = 2

model = "cnn"           # What NN we'll be using


print("Loading known faces...")

knownFaces = []         # List of all the different faces that we know
knownNames = []         # Give a name to each face

for name in os.listdir(DIR_KNOWN):                      # List of all the names in the Known directory
    for filename in os.listdir(f"{DIR_KNOWN}/{name}"):  # List all files there
        # Load an image
        image = face_recognition.load_image_file(f"{DIR_KNOWN}/{name}/{filename}")
        
        # Encode only one face in that image (hence the [0])
        # Known faces should only have one face in it
        encoding = face_recognition.face_encodings(image)[0]

        # Add the face and the name to the lists
        knownFaces.append(encoding)
        knownNames.append(name)


