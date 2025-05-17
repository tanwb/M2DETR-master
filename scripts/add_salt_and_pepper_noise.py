import os
from PIL import Image
import numpy as np
import random


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    """
    Add salt-and-pepper noise to a PIL Image object.

    Salt noise and pepper noise are randomly
    applied to the input image based on specified probabilities.

    Parameters:
    image (PIL.Image): Input image in PIL format
    salt_prob (float): Probability of adding salt noise (range: 0-1)
    pepper_prob (float): Probability of adding pepper noise (range: 0-1)

    Returns:
    PIL.Image: Modified image with noise
    """
    image_array = np.array(image)

    num_salt = np.ceil(salt_prob * image_array.size)
    num_pepper = np.ceil(pepper_prob * image_array.size)

    # add salt
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image_array.shape]
    image_array[tuple(coords)] = 255  # Changed from 1 to 255 for proper white pixels

    # add pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image_array.shape]
    image_array[tuple(coords)] = 0    # 0 remains for black pixels

    return Image.fromarray(image_array)


def process_images_in_folder_train(folder_path_train_in, folder_path_train_out, salt_prob, pepper_prob):
    """
    Process all images in the specified folder by adding salt-and-pepper noise.
    Args:
    salt_prob (float): Probability of adding salt noise [0-1]
    pepper_prob (float): Probability of adding pepper noise [0-1]
    """
    if not os.path.exists(folder_path_train_out):
        os.makedirs(folder_path_train_out)

    for filename in os.listdir(folder_path_train_in):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(folder_path_train_in, filename)
            image = Image.open(image_path)

            noisy_image = add_salt_and_pepper_noise(image, salt_prob, pepper_prob)

            noisy_image.save(os.path.join(folder_path_train_out, f'{filename}'))


def process_images_in_folder_test(folder_path_test_in, folder_path_test_out, salt_prob, pepper_prob):
    """
    Process all images in the specified folder by adding salt-and-pepper noise.
    Args:
    salt_prob (float): Probability of adding salt noise [0-1]
    pepper_prob (float): Probability of adding pepper noise [0-1]
    """
    if not os.path.exists(folder_path_test_out):
        os.makedirs(folder_path_test_out)

    for filename in os.listdir(folder_path_test_in):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            image_path = os.path.join(folder_path_test_in, filename)
            image = Image.open(image_path)

            noisy_image = add_salt_and_pepper_noise(image, salt_prob, pepper_prob)

            noisy_image.save(os.path.join(folder_path_test_out, f'{filename}'))


if __name__ == '__main__':
    folder_path_train_in = 'XXX/data_ip102/coco_format/train'
    folder_path_train_out = 'XXX/data_ip102/coco_format/noise/train_add_salt_pepper_noise'

    folder_path_test_in = 'XXX/data_ip102/coco_format/test'
    folder_path_test_out = 'XXX/data_ip102/coco_format/noise/test_add_salt_pepper_noise'

    salt_prob = 0.03  # Probability for salt noise
    pepper_prob = 0.03  # Probability for pepper noise

    # Process all images in the target folder
    process_images_in_folder_train(folder_path_train_in, folder_path_train_out,
                                 salt_prob=salt_prob, pepper_prob=pepper_prob)
    process_images_in_folder_test(folder_path_test_in, folder_path_test_out,
                               salt_prob=salt_prob, pepper_prob=pepper_prob)
