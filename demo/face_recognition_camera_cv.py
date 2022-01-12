import os
import sys

import cv2
import face_recognition
import numpy as np
from numpy.linalg import norm
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.predict_arcface import encode_sample


def get_distances(known_face_encodings, face_encoding):
    return face_recognition.face_distance(known_face_encodings, face_encoding)


def arc_cos(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def compare_arc(known_face_encodings, face_encoding):
    res = []
    for fe in known_face_encodings:
        res.append(arc_cos(fe, face_encoding))
    return np.array(res)


def softmax_encode(image, known_face_locations=None):
    return face_recognition.face_encodings(image, known_face_locations=known_face_locations)


def get_boundaries(locations):
    return (min(locations[3], locations[1]),
            max(locations[3], locations[1]),
            min(locations[2], locations[0]),
            max(locations[2], locations[0]))


def arcface_encode(image, known_face_locations=None):
    if known_face_locations is not None and len(known_face_locations) > 0:
        ll, rr, dd, uu = get_boundaries(known_face_locations[0])
        image = image[max(0, dd):uu, max(0, ll):rr]
    return encode_sample(wht="../models/model.pth", img=image)


def run_demo(tolerance=0.6,
             encoding_method=softmax_encode,
             distances_method=get_distances,
             method='soft'):
    video_capture = cv2.VideoCapture(0)

    obama_image = face_recognition.load_image_file("images/obama.jpg")
    obama_face_encoding = encoding_method(obama_image)[0]

    biden_image = face_recognition.load_image_file("images/biden.jpg")
    biden_face_encoding = encoding_method(biden_image)[0]

    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding,
    ]
    known_face_names = [
        "Barack Obama",
        "Joe Biden",
    ]

    face_locations = []
    face_names = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
            face_encodings = encoding_method(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                face_distances = distances_method(known_face_encodings, face_encoding)
                if method == 'arc':
                    best_match_index = np.argmax(face_distances)
                    name = known_face_names[best_match_index] if max(face_distances) > tolerance else "Unknown"
                else:
                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index] if min(face_distances) < tolerance else "Unknown"
                face_names.append(name)

        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    desc = 'Basic script to demonstrate face recognition. Press `q` to quit. Add images to folder images and add them to code'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--method', type=str, default='softmax', help="arc or softmax")
    parser.add_argument('--tolerance', type=float, default=0.2)

    args = parser.parse_args()

    if args.method == 'arc':
        run_demo(method='arc', encoding_method=arcface_encode, distances_method=compare_arc)
    else:
        run_demo()
