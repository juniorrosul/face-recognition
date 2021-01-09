import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

image = face_recognition.load_image_file('/home/madsonjr/repositories/face-recognition/src/madson_3.jpg')
encoding_image = face_recognition.face_encodings(image)

know_faces = [
    encoding_image[0]
]

know_names = [
    'Madson'
]

while True:
    ret, frame = video_capture.read()

    # Convert BGR para RGB
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, botton, left), face_encodings in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(know_faces, face_encodings)
        name = 'Desconhecido'

        face_distances = face_recognition.face_distance(know_faces, face_encodings)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = know_names[best_match_index]
            
        cv2.rectangle(frame, (left, top), (right, botton), (255, 0, 0), 2)

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, name, (left+6, botton-6), font, 0.5, (255, 0, 0), 1)

    cv2.imshow('eSales - Face recognition', frame)

    if cv2.waitKey(1) & 0xFF == 'q':
        break


video_capture.release()
cv2.destroyAllWindows()