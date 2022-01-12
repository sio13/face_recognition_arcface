import argparse
import glob
import os
import pickle

import cv2
import face_recognition
import torch

from models.predict_arcface import encode_sample


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--train_data", required=True,
                            help="path to train dataset - must contain directory <name> and PNGs in directories")
    arg_parser.add_argument("--model_path", required=True, help="path to model - will be rewritten")
    arg_parser.add_argument("--method", type=str, default="cnn",
                            help="cnn for convolution neural network and hog for histogram of oriented gradients")
    return vars(arg_parser.parse_args())


def get_train_dirs_paths(train_data_path):
    train_data_path = os.path.abspath(train_data_path)
    return [sample for sample in glob.glob(f"{train_data_path}/**")]


def encode_image_fr_framework(path_to_sample, image_path, model='cnn'):
    image = cv2.imread(os.path.abspath(f"{path_to_sample}/{image_path}"))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=model)
    return face_recognition.face_encodings(rgb, boxes)


def encode_image_arc_face(path_to_sample, image_path, weights="models/backbone.pth"):
    wht = torch.load(weights, map_location=torch.device('cpu'))
    return encode_sample(wht, os.path.abspath(f"{path_to_sample}/{image_path}"))


def process_sample(path_to_sample, model, image_encodings, image_names, method):
    for image_path in os.listdir(path_to_sample):
        print(f"Processing {image_path}")

        sample_name = path_to_sample.split(os.path.sep)[-1]
        if method == 'arc':
            encodings = encode_image_arc_face(path_to_sample, image_path, weights="models/backbone.pth")
        else:
            encodings = encode_image_fr_framework(path_to_sample, image_path, model)
        for encoding in encodings:
            image_encodings.append(encoding)
            image_names.append(sample_name)


def encode_samples(train_data, model, image_encodings, image_names, method):
    for sample in get_train_dirs_paths(train_data):
        process_sample(sample, model, image_encodings, image_names, method=method)


def create_model(path_to_model, train_data, model, method='arc'):
    image_encodings = []
    image_names = []
    encode_samples(train_data, model, image_encodings, image_names, method=method)
    print(f"Saving model to {path_to_model}")
    model_file = open(path_to_model, "wb")
    model_file.write(pickle.dumps({"encodings": image_encodings, "names": image_names}))
    model_file.close()


if __name__ == '__main__':
    arguments = get_args()
    create_model(arguments["model_path"], arguments["train_data"], arguments["method"], method='arc')
