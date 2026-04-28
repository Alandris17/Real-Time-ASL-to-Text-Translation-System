import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

def rotate_image(image, angle, scale=1.0):
    rows, columns = image.shape[:2]
    center = (columns / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (columns, rows), borderMode=cv2.BORDER_REFLECT_101)

def flip_images(dataset_path):
    """
    Loads images from the dataset folder (expects structure: dataset_path/LabelName/image.jpg)
    Extracts hand landmarks for each image.
    Returns:
      - X: A NumPy array of landmark feature vectors.
      - y: A NumPy array of corresponding labels.
      - label_list: The list of label names.
    """
    counter = 0
    label_list = []
    for label in os.listdir(dataset_path):
        label_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(label_dir):
            print(label_dir, "is not a directory")
            continue
        label_list.append(label)
        print("Loading label:", label)
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            counter += 1
            image = cv2.imread(file_path)
            if image is None:
                print("Invalid image:", file_path)
                continue
            flipped_image = cv2.flip(image, 1)
            cv2.imwrite(f'{label_dir}/{file[:-4]}_flipped.jpg',flipped_image)
            for angle in range (-30, 30, 5):
                rotated_normal = rotate_image(image, angle=angle, scale=1.1)
                cv2.imwrite(f'{label_dir}/{file[:-4]}_rotated_{angle}.jpg',rotated_normal)
                rotated_flipped = rotate_image(flipped_image, angle=angle, scale = 1.1)
                cv2.imwrite(f'{label_dir}/{file[:-4]}_flipped_rotated_{angle}.jpg',rotated_flipped)


dataset_path = "Data_preprocessed"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

flip_images(dataset_path)