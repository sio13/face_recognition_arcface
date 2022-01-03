import argparse
import os

import cv2

from align_face import process_face


def process_folder(category, raw_data_dir, target_dir):
    global_path_to_category = os.path.join(raw_data_dir, category)
    for image_path in os.listdir(global_path_to_category):
        print(f"Processing {image_path} file")

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        if not os.path.isdir(os.path.join(target_dir, category)):
            os.makedirs(os.path.join(target_dir, category))

        global_image_path = os.path.join(raw_data_dir, category, image_path)
        result = process_face(global_image_path)

        target_image_path = os.path.join(target_dir, category, image_path)
        cv2.imwrite(target_image_path, result)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--raw_data", required=True, type=str,
                            help="path to raw dataset - must be in the correct structure (root and names of labels)")
    arg_parser.add_argument("--processed_data", required=True, type=str)
    args = arg_parser.parse_args()

    for category in os.listdir(args.raw_data):
        print(category)
        process_folder(category,
                       args.raw_data,
                       args.processed_data)


if __name__ == '__main__':
    main()
