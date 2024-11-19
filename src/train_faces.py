import argparse
import cv2
from face_recognition.train import train_recognizer, save_recognizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train the face recognition model.")
    parser.add_argument('--train_dir', type=str, default="../resources/data/faces",
                        help='Path of the train directory')
    parser.add_argument('--model_path', type=str, default="../resources/models/faces/faces",
                        help='Path where the trained model should be saved')

    args = parser.parse_args()

    # Split model_path to get the base path and model name
    model_base_path = "/".join(args.model_path.split("/")[:-1])
    model_name = args.model_path.split("/")[-1]

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer, labels_name_map = train_recognizer(recognizer, args.train_dir)

    # Use 'model_name' instead of 'args.model_name'
    save_recognizer(recognizer, labels_name_map, recognizer_name=model_name,
                    base_path=model_base_path)
