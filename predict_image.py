import argparse
import os
import pickle

import cv2
import face_recognition
import torch

from models.predict_arcface import encode_sample


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_path", required=True,
                            help="path to model")
    arg_parser.add_argument("--image", required=True,
                            help="path to image to predict")
    arg_parser.add_argument("--method", type=str, default="cnn",
                            help="cnn for convolution neural network and hog for histogram of oriented gradients")
    arg_parser.add_argument("--tolerance", type=float, default=0.6,
                            help="setting tolerance for cnn and hog, default is 0.6")
    return vars(arg_parser.parse_args())


def load_model(path_to_model):
    return pickle.loads(open(path_to_model, "rb").read())


def extract_face_fr_framework(image_path, model):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_grid = face_recognition.face_locations(rgb, model=model)
    face_encodings = face_recognition.face_encodings(rgb, face_grid)
    if len(face_encodings) != 1:
        print("Expected just one face image!")
    try:
        return face_encodings[0]
    except Exception as e:
        print(f"Exception occurred {str(e)}")
        return None


def extract_arc_face(image_path, weights="models/backbone.pth"):
    wht = torch.load(weights, map_location=torch.device('cpu'))
    return encode_sample(wht, os.path.abspath(f"{image_path}"))


def detect_face(image_path, model, tolerance, model_vector, method):
    if method == 'arc':
        encoded_face = extract_arc_face(image_path)
    else:
        encoded_face = extract_face_fr_framework(image_path, model)
    if encoded_face is None:
        print("Nothing found!")
        return {}
    vectors_in_tolerance = face_recognition.compare_faces(model_vector["encodings"], encoded_face, tolerance=19)
    matched_vectors = [i for (i, b) in enumerate(vectors_in_tolerance) if b]
    counts = {}
    for i in matched_vectors:
        name = model_vector["names"][i]
        counts[name] = counts.get(name, 0) + 1
    return counts


def get_face_probability_function(image_path, model, tolerance, model_vector, method):
    results = detect_face(image_path, model, tolerance, model_vector, method=method)

    total_pieces = sum([value for value in results.values()])
    return {key: float(value) / total_pieces for key, value in results.items()}


if __name__ == '__main__':
    arguments = get_args()
    model_vector_ = load_model(arguments["model_path"])
    prob_dict = get_face_probability_function(arguments["image"],
                                              arguments["method"],
                                              float(arguments["tolerance"]),
                                              model_vector_, method='arc')
    print(max(prob_dict, key=prob_dict.get))
