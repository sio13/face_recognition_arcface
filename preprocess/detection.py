import copy

import cv2
import face_recognition


class NoFaceException(Exception):
    pass


def get_boundaries(locations):
    return (min(locations[3], locations[1]),
            max(locations[3], locations[1]),
            min(locations[2], locations[0]),
            max(locations[2], locations[0]))


def get_locations(image):
    """
    This method rotates the photo until it finds the face position.
    """
    face_locations = None
    gray = None
    for current_angle in range(0, 340, 5):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -current_angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

        face_locations = face_recognition.face_locations(gray, model='cnn')
        if len(face_locations) >= 1:
            image = rotated
            break
    if face_locations is not None and gray is not None:
        return face_locations, image, gray
    else:
        raise NoFaceException("No face found!")


def show_detected_face(image, model='cnn'):
    image = copy.deepcopy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(gray, model=model)
    cv2.rectangle(image, (face_locations[0][1], face_locations[0][0]), (face_locations[0][3], face_locations[0][2]),
                  (255, 0, 0), 2)
    while True:
        cv2.imshow("img", image)
        cv2.waitKey(30)
