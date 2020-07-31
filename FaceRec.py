import face_recognition
import os
import cv2


DIR_KNOWN = "Known"     # Directories where the faces are
DIR_UNKNOWN = "Unknown"

tolerance = 0.6         # Tolerance of a match (% of accuracy permitted)

frameThickness = 3      # These two are to draw rectangles around the faces
fontThickness = 2

model = "cnn"           # What NN we'll be using

knownFaces = []         # List of all the different faces that we know
knownNames = []         # Give a name to each face


print("Loading known faces...")

for name in os.listdir(DIR_KNOWN):                      # List of all the names in the Known directory
    for filename in os.listdir(f"{DIR_KNOWN}/{name}"):  # List all files there
        print("Checking " + filename)
        # Load an image
        image = face_recognition.load_image_file(f"{DIR_KNOWN}/{name}/{filename}")
        
        # If we found exactly one face, use it
        if(len(face_recognition.face_encodings(image)) == 1):
            encoding = face_recognition.face_encodings(image)

            # Add the face and the name to the lists
            knownFaces.append(encoding)
            knownNames.append(name)

        elif(len(face_recognition.face_encodings(image)) > 1):
            print(" File " + filename + " had more than one face")

        elif(len(face_recognition.face_encodings(image)) < 1):
            print(" File " + filename + " had no faces")


print("Processing unknown faces...")

for filename in os.listdir(DIR_UNKNOWN):
    print("Checking " + filename)
    image = face_recognition.load_image_file(f"{DIR_UNKNOWN}/{filename}")

    # Find every face in each given image and encode each one
    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, locations)

    # Now to draw on the image, we need to make it workable by CV2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # For each encoding that we have, compare against encodings of known faces
    for faceEncoding, faceLocation in zip(encodings, locations):
        results = face_recognition.compare_faces(knownFaces, faceEncoding, tolerance)
        print(results)

        # If we have a match, the associated name will be in the same position as the face
        # was in knownFaces. We'll have to handle the array "results" in a way that we can
        # work with (results is an array of arrays)
        index = -1
        
        # subdivision = second layer array we're at
        for subdivision in results:
            isMatch = 1
            index += 1

            for result in subdivision:
                if result == False:     # At least one false and it's not a match
                    isMatch = 0
                    break

            if (isMatch == 0):
                break
    
            match = knownNames[index]
            print("Match found: " + match)

            # Draw a rectangle around the face
            topLeft = (faceLocation[3], faceLocation[0])
            bottomRight = (faceLocation[1], faceLocation[2])

            # Have a specific color for each name
            color = [(ord(c.lower())-97)*8 for c in match[:3]]

            cv2.rectangle(image, topLeft, bottomRight, color, frameThickness)

            # Nameplate
            topLeft = (faceLocation[3], faceLocation[2])
            bottomRight = (faceLocation[1], faceLocation[2] + 22)

            cv2.rectangle(image, topLeft, bottomRight, color, cv2.FILLED)

            # Name the rectangle
            cv2.putText(image, match, (faceLocation[3] + 10, faceLocation[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), fontThickness)

            break


    # Show the image until I press something
    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)