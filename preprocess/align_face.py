import copy
import os

import cv2
import face_recognition
import numpy as np

from detection import get_locations, get_boundaries


def get_eyes_positions(gray_face, tries=10, initial_coef=38):
    """
    This method find eyes positions on this photo and returns their position.
    We expect two eyes thus we will adapt sensitivity to find two eyes, if not we'll return None
    """
    eye_cascade = cv2.CascadeClassifier(os.path.join("models", "haarcascade_eye.xml"))
    if not os.path.exists(os.path.join("models", "haarcascade_eye.xml")):
        raise Exception("haarcascade model cannot be opened!")
    initial_eyes_index = initial_coef
    for counter in range(tries):
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, initial_eyes_index)
        if len(eyes) != 2:
            if len(eyes) < 2:
                initial_eyes_index -= 2
            else:
                initial_eyes_index += 2
        else:
            return eyes
    return None


def show_eyes_position(image, eyes):
    image = copy.deepcopy(image)
    left_eye = eyes[0]
    right_eye = eyes[1]
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))

    cv2.circle(image, left_eye_center, 5, (255, 0, 0), -1)
    cv2.circle(image, right_eye_center, 5, (255, 0, 0), -1)
    cv2.line(image, right_eye_center, left_eye_center, (0, 200, 200), 3)
    while True:
        cv2.imshow("img", image)
        cv2.waitKey(30)


def process_face(image_path):
    """
    This function detects the face locations a align it using eyes locations to the average position.
    """
    image = cv2.imread(image_path)
    face_locations, image, gray = get_locations(image)

    for i in range(len(face_locations[:1])):
        ll, rr, dd, uu = get_boundaries(face_locations[0])
        margin = 70

        gray_face = gray[max(0, dd - margin):uu + margin, max(0, ll - margin):rr + margin]
        clipped_face = image[max(0, dd - margin):uu + margin, max(0, ll - margin):rr + margin]

        eyes = get_eyes_positions(gray_face)
        if eyes is None:
            return clipped_face
        eye_1 = eyes[0]
        eye_2 = eyes[1]

        if eye_1[0] < eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            left_eye = eye_2
            right_eye = eye_1

        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y

        angle = np.arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi

        h, w = clipped_face.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(clipped_face, rotation_matrix, (w, h))

        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        rotated_face_locations = face_recognition.face_locations(gray_rotated, model='cnn')
        if len(rotated_face_locations) == 0:
            return clipped_face
        rll, rrr, rdd, ruu = get_boundaries(rotated_face_locations[0])
        margin_rot = 25
        result = rotated[max(0, rdd - margin_rot):ruu + margin_rot, max(0, rll - margin_rot):rrr + margin_rot]
        return result


if __name__ == '__main__':
    pass
    # im = cv2.imread("/Users/martingano/repos/pova_projekt/full_pipeline/jarka.jpg")
    # face_locations, image, gray = get_locations(im)
    # ll, rr, dd, uu = get_boundaries(face_locations[0])
    # gray_face = gray[max(0, dd):uu, max(0, ll):rr]
    # show_eyes_position(gray_face, get_eyes_positions(gray_face))
    # show_detected_face(cv2.imread("/Users/martingano/repos/pova_projekt/full_pipeline/jarka.jpg"))
